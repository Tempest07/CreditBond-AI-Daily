from __future__ import annotations

import json
import re
import html
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import prepare_wide_data, read_csv_auto
from .dm_api import (
    curve_data_to_edb_like_raw,
    edb_raw_to_point_in_time_wide,
    fetch_bond_yield_curve_data,
    fetch_edb_indicator_data,
    parse_dm_datetime,
)
from .predict import predict_latest


@dataclass
class DailyUpdatePaths:
    raw_dir: Path
    processed_dir: Path
    report_dir: Path
    wide_path: Path
    dictionary_path: Path
    model_ready_path: Path
    features_path: Path
    json_report_path: Path
    markdown_report_path: Path
    html_report_path: Path


def _is_enabled(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text not in {"0", "false", "否", "no", "n", "disabled", "停用"}


def _safe_name(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", str(value))
    return re.sub(r"_+", "_", text).strip("_") or "model"


def _cell_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _canonical_dm_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "indicator_id": {"indicator_id", "indicatorId"},
        "indicator_name": {"indicator_name", "indicatorName"},
        "data_date": {"data_date", "dataDate"},
        "data_value": {"data_value", "dataValue"},
        "basic_indicator_unit": {"basic_indicator_unit", "basicIndicatorUnit"},
        "publish_time": {"publish_time", "publishTime"},
        "data_source": {"data_source", "dataSource"},
        "statistical_frequency": {"statistical_frequency", "statisticalFrequency"},
    }
    mapping = {}
    for col in df.columns:
        for canonical, names in aliases.items():
            if str(col) in names:
                mapping[str(col)] = canonical
                break
    return df.rename(columns=mapping)


def load_dm_indicator_config(path: str | Path) -> pd.DataFrame:
    config = read_csv_auto(path)
    if "enabled" in config.columns:
        config = config[config["enabled"].map(_is_enabled)].copy()
    if "indicator_id" not in config.columns:
        raise ValueError("DM指标配置必须包含 indicator_id 列。")
    config["indicator_id"] = config["indicator_id"].astype(str).str.strip()
    config = config[config["indicator_id"].ne("") & config["indicator_id"].str.lower().ne("nan")].copy()
    if config.empty:
        raise ValueError("DM指标配置没有可用的 indicator_id。")
    return config.reset_index(drop=True)


def _existing_max_data_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        existing = _canonical_dm_columns(pd.read_csv(path, encoding="utf-8-sig"))
    except Exception:
        return None
    if "data_date" not in existing.columns or existing.empty:
        return None
    parsed = parse_dm_datetime(existing["data_date"]).dropna()
    if parsed.empty:
        return None
    return parsed.max().date().isoformat()


def _merge_and_save_raw(
    existing_path: Path,
    new_data: pd.DataFrame,
    alias: str | None = None,
    replace_existing: bool = False,
) -> int:
    frames = []
    if existing_path.exists() and not replace_existing:
        existing = _canonical_dm_columns(pd.read_csv(existing_path, encoding="utf-8-sig"))
        existing["_merge_source_order"] = 0
        frames.append(existing)
    if not new_data.empty:
        frame = _canonical_dm_columns(new_data.copy())
        if alias:
            frame["alias"] = alias
        frame["_merge_source_order"] = 1
        frames.append(frame)
    if not frames:
        if replace_existing and existing_path.exists():
            existing_path.unlink()
        return 0
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _canonical_dm_columns(combined)
    if "data_date" in combined.columns:
        combined["_data_date_parsed"] = parse_dm_datetime(combined["data_date"])
    else:
        combined["_data_date_parsed"] = pd.NaT
    subset = [col for col in ["indicator_id", "_data_date_parsed"] if col in combined.columns]
    if subset:
        combined = combined.sort_values(subset + ["_merge_source_order"]).drop_duplicates(subset=subset, keep="last")
    combined = combined.drop(columns=[c for c in ["_data_date_parsed", "_merge_source_order"] if c in combined.columns])
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(existing_path, index=False, encoding="utf-8-sig")
    return int(len(combined))


def _row_source(row: pd.Series) -> str:
    return (_cell_text(row.get("source", "")) or "edb").strip().lower()


def _is_curve_source(source: str) -> bool:
    return source in {"curve", "curve_func", "yield_curve", "bond_yield_curve", "bond_yield_curve_func"}


def _row_fetch_start(
    row: pd.Series,
    output_file: Path,
    default_start_date: str | None,
    incremental: bool,
    overlap_days: int,
) -> str:
    config_start = _cell_text(row.get("start_date", ""))
    start = default_start_date or config_start
    existing_max = _existing_max_data_date(output_file)
    if incremental and existing_max:
        start_ts = pd.Timestamp(existing_max) - pd.Timedelta(days=overlap_days)
        start = start_ts.date().isoformat()
    if not start:
        indicator_id = _cell_text(row.get("indicator_id", ""))
        raise ValueError(f"{indicator_id} 缺少 start_date，且本地没有历史原始文件。")
    return start


def _chunked(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def fetch_dm_raw_incremental(
    config: pd.DataFrame,
    raw_dir: str | Path,
    default_start_date: str | None,
    end_date: str,
    incremental: bool = True,
    overlap_days: int = 15,
    base_url: str | None = None,
    timeout: int = 30,
) -> list[dict]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curve_rows = []
    for _, row in config.iterrows():
        indicator_id = str(row["indicator_id"]).strip()
        alias = _cell_text(row.get("alias", ""))
        frequency = _cell_text(row.get("frequency", "")) or _cell_text(row.get("statistical_frequency", "")) or None
        output_file = raw_dir / f"{indicator_id}.csv"
        source = _row_source(row)

        if _is_curve_source(source):
            curve_name = _cell_text(row.get("curve_name", ""))
            curve_term = _cell_text(row.get("curve_term", "")) or _cell_text(row.get("curve_terms", ""))
            if not curve_name or not curve_term:
                raise ValueError(f"{indicator_id} 使用 curve_func 时必须配置 curve_name 和 curve_term。")
            curve_rows.append(
                {
                    "row": row,
                    "indicator_id": indicator_id,
                    "alias": alias,
                    "frequency": frequency or "日",
                    "output_file": output_file,
                    "fetch_start": _row_fetch_start(row, output_file, default_start_date, incremental, overlap_days),
                    "curve_data_source": _cell_text(row.get("curve_data_source", ""))
                    or _cell_text(row.get("data_source", ""))
                    or "18",
                    "curve_name": curve_name,
                    "curve_term": curve_term,
                    "curve_type": _cell_text(row.get("curve_type", "")) or "1",
                }
            )
            continue

        start = _row_fetch_start(row, output_file, default_start_date, incremental, overlap_days)

        data = fetch_edb_indicator_data(
            indicator_id=indicator_id,
            start_date=start,
            end_date=end_date,
            frequency=frequency,
            base_url=base_url,
            timeout=timeout,
        )
        total_rows = _merge_and_save_raw(output_file, data, alias=alias, replace_existing=not incremental)
        rows.append(
            {
                "indicator_id": indicator_id,
                "alias": alias,
                "frequency": frequency or "",
                "fetch_start": start,
                "fetch_end": end_date,
                "new_rows": int(len(data)),
                "total_rows": total_rows,
                "output_file": str(output_file),
            }
        )
    curve_groups: dict[tuple[str, str, str], list[dict]] = {}
    for item in curve_rows:
        key = (str(item["curve_data_source"]), str(item["curve_name"]), str(item["curve_type"]))
        curve_groups.setdefault(key, []).append(item)

    for (curve_data_source, curve_name, curve_type), group in curve_groups.items():
        terms = sorted({str(item["curve_term"]) for item in group}, key=lambda value: float(value))
        by_term: dict[str, list[dict]] = {}
        for item in group:
            by_term.setdefault(str(item["curve_term"]), []).append(item)
        for term_chunk in _chunked(terms, 5):
            fetch_start = min(item["fetch_start"] for term in term_chunk for item in by_term[term])
            data = fetch_bond_yield_curve_data(
                data_source=curve_data_source,
                curve_name=curve_name,
                curve_terms=term_chunk,
                curve_type=curve_type,
                start_date=fetch_start,
                end_date=end_date,
                base_url=base_url,
                timeout=timeout,
            )
            for term in term_chunk:
                for item in by_term[term]:
                    raw_like = curve_data_to_edb_like_raw(
                        data,
                        indicator_id=item["indicator_id"],
                        alias=item["alias"],
                        curve_term=item["curve_term"],
                        frequency=item["frequency"],
                    )
                    total_rows = _merge_and_save_raw(
                        item["output_file"],
                        raw_like,
                        alias=item["alias"],
                        replace_existing=not incremental,
                    )
                    rows.append(
                        {
                            "indicator_id": item["indicator_id"],
                            "alias": item["alias"],
                            "frequency": item["frequency"],
                            "source": "curve_func",
                            "curve_data_source": curve_data_source,
                            "curve_name": curve_name,
                            "curve_term": item["curve_term"],
                            "curve_type": curve_type,
                            "fetch_start": fetch_start,
                            "fetch_end": end_date,
                            "new_rows": int(len(raw_like)),
                            "total_rows": total_rows,
                            "output_file": str(item["output_file"]),
                        }
                    )
    return rows


def build_daily_paths(
    out_dir: str | Path,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> DailyUpdatePaths:
    out_dir = Path(out_dir)
    raw = Path(raw_dir) if raw_dir else out_dir / "raw"
    processed = Path(processed_dir) if processed_dir else out_dir / "processed"
    reports = Path(report_dir) if report_dir else out_dir / "reports"
    return DailyUpdatePaths(
        raw_dir=raw,
        processed_dir=processed,
        report_dir=reports,
        wide_path=processed / "dm_wide_latest.csv",
        dictionary_path=processed / "dm_dictionary_latest.csv",
        model_ready_path=processed / "dm_model_ready_latest.csv",
        features_path=processed / "dm_features_latest.csv",
        json_report_path=reports / "daily_dm_update_report.json",
        markdown_report_path=reports / "daily_dm_update_report.md",
        html_report_path=reports / "daily_dm_update_report.html",
    )


def find_model_dirs(model_dirs: Iterable[str] | None = None, models_root: str | Path | None = None) -> list[Path]:
    found: list[Path] = []
    for item in model_dirs or []:
        if not item:
            continue
        path = Path(item)
        if (path / "model.pt").exists():
            found.append(path)
    if models_root:
        root = Path(models_root)
        if root.exists():
            for model_file in sorted(root.rglob("model.pt")):
                found.append(model_file.parent)
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _summarize_prediction(result: dict) -> str:
    probs = result.get("probabilities", {})
    prob_text = "，".join(f"{k} {float(v):.2%}" for k, v in probs.items())
    return f"{result.get('prediction', '')}（{prob_text}）"


def _valid_predictions(report: dict) -> list[dict]:
    return [item["result"] for item in report.get("predictions", []) if item.get("ok") and item.get("result")]


def _average_probabilities(predictions: list[dict]) -> dict[str, float]:
    labels = ["看空", "看多", "震荡"]
    if not predictions:
        return {label: 0.0 for label in labels}
    return {
        label: float(np.mean([item.get("probabilities", {}).get(label, 0.0) for item in predictions]))
        for label in labels
    }


DAILY_SIGNAL_PROB_THRESHOLD = 0.45
DAILY_SIGNAL_MARGIN_VS_RANGE = 0.03
DAILY_DISPLAY_TILT_THRESHOLD = 0.40
DAILY_DISPLAY_STRONG_THRESHOLD = 0.60
LABELS_CN_DISPLAY = ["看空", "看多", "震荡"]


def _prob_vector(result: dict) -> np.ndarray:
    probs = result.get("probabilities", {})
    if isinstance(probs, dict) and len(probs) >= 3:
        values = list(probs.values())[:3]
        return np.asarray([float(v) for v in values], dtype=float)
    return np.zeros(3, dtype=float)


def _prob_dict_from_vector(values: np.ndarray) -> dict[str, float]:
    return {LABELS_CN_DISPLAY[i]: float(values[i]) for i in range(3)}


def _display_signal_from_threshold(
    threshold: dict,
    tilt_threshold: float = DAILY_DISPLAY_TILT_THRESHOLD,
    strong_threshold: float = DAILY_DISPLAY_STRONG_THRESHOLD,
) -> dict:
    direction = str(threshold.get("direction_prediction", ""))
    direction_prob = float(threshold.get("direction_prob", 0.0))
    if direction not in {"看空", "看多"} or direction_prob < tilt_threshold:
        display_prediction = "震荡"
        display_direction = "震荡"
        display_strength = "信号弱"
    elif direction_prob >= strong_threshold:
        display_prediction = f"强{direction}"
        display_direction = direction
        display_strength = "强信号"
    else:
        display_prediction = "震荡偏多" if direction == "看多" else "震荡偏空"
        display_direction = direction
        display_strength = "倾向信号"
    return {
        "display_prediction": display_prediction,
        "display_direction": display_direction,
        "display_strength": display_strength,
        "display_tilt_threshold": tilt_threshold,
        "display_strong_threshold": strong_threshold,
    }


def _threshold_signal_from_probs(
    probs: np.ndarray,
    min_action_prob: float = DAILY_SIGNAL_PROB_THRESHOLD,
    min_margin_vs_range: float = DAILY_SIGNAL_MARGIN_VS_RANGE,
) -> dict:
    bearish, bullish, range_prob = [float(x) for x in probs]
    raw_label = int(np.argmax(probs))
    if bullish >= bearish:
        direction_label = 1
        direction_prob = bullish
    else:
        direction_label = 0
        direction_prob = bearish
    margin_vs_range = direction_prob - range_prob
    active = direction_prob >= min_action_prob and margin_vs_range >= min_margin_vs_range
    final_label = direction_label if active else 2
    threshold = {
        "raw_label": raw_label,
        "raw_prediction": LABELS_CN_DISPLAY[raw_label],
        "threshold_label": final_label,
        "threshold_prediction": LABELS_CN_DISPLAY[final_label],
        "is_active": bool(active),
        "status": "有效信号" if active else "信号弱",
        "direction_label": direction_label,
        "direction_prediction": LABELS_CN_DISPLAY[direction_label],
        "direction_prob": direction_prob,
        "range_prob": range_prob,
        "margin_vs_range": margin_vs_range,
        "prob_threshold": min_action_prob,
        "margin_threshold": min_margin_vs_range,
    }
    threshold.update(_display_signal_from_threshold(threshold))
    return threshold


def build_ensemble_predictions(predictions: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for item in predictions:
        target = str(item.get("target_col", ""))
        if target:
            grouped.setdefault(target, []).append(item)

    ensembles: dict[str, dict] = {}
    for target, items in grouped.items():
        vectors = [_prob_vector(item) for item in items]
        vectors = [vec for vec in vectors if len(vec) == 3 and np.isfinite(vec).all()]
        if not vectors:
            continue
        avg_probs = np.mean(np.vstack(vectors), axis=0)
        threshold = _threshold_signal_from_probs(avg_probs)
        ensembles[target] = {
            "target_col": target,
            "model_count": len(vectors),
            "model_dirs": [str(item.get("model_dir", "")) for item in items],
            "probabilities": _prob_dict_from_vector(avg_probs),
            **threshold,
        }
    return ensembles


def _model_history_stats(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    metrics_path = model_dir / "metrics.json"
    predictions_path = model_dir / "test_predictions.csv"
    stats = {
        "has_history": False,
        "accuracy": None,
        "macro_f1": None,
        "total_return_proxy": None,
        "active_signal_ratio": None,
        "test_start": "",
        "test_end": "",
        "test_count": 0,
    }
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            report = metrics.get("classification_report", {})
            backtest = metrics.get("backtest", {})
            stats.update(
                {
                    "has_history": True,
                    "accuracy": report.get("accuracy"),
                    "macro_f1": report.get("macro avg", {}).get("f1-score"),
                    "total_return_proxy": backtest.get("total_return_proxy"),
                    "active_signal_ratio": backtest.get("active_signal_ratio"),
                }
            )
        except Exception:
            pass
    if predictions_path.exists():
        try:
            pred = pd.read_csv(predictions_path, encoding="utf-8-sig")
            if "date" in pred.columns and not pred.empty:
                dates = pd.to_datetime(pred["date"])
                stats.update(
                    {
                        "has_history": True,
                        "test_start": dates.min().strftime("%Y-%m-%d"),
                        "test_end": dates.max().strftime("%Y-%m-%d"),
                        "test_count": int(len(pred)),
                    }
                )
        except Exception:
            pass
    return stats


def _series_points(df: pd.DataFrame, col: str, n: int = 20) -> list[dict]:
    if col not in df.columns:
        return []
    sub = df[["date", col]].dropna().tail(n).copy()
    return [
        {"date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"), "value": float(row[col])}
        for _, row in sub.iterrows()
    ]


def _bp_change(points: list[dict], periods: int) -> float | None:
    if len(points) <= periods:
        return None
    return (float(points[-1]["value"]) - float(points[-1 - periods]["value"])) * 100.0


def _raw_change(points: list[dict], periods: int) -> float | None:
    if len(points) <= periods:
        return None
    return float(points[-1]["value"]) - float(points[-1 - periods]["value"])


def _latest_value(points: list[dict]) -> float | None:
    return float(points[-1]["value"]) if points else None


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def build_market_snapshot(features: pd.DataFrame, predictions: list[dict]) -> dict:
    columns = list(features.columns)
    prediction_by_target = {}
    for item in predictions:
        target = str(item.get("target_col", ""))
        if target and target not in prediction_by_target:
            prediction_by_target[target] = item
    ensemble_by_target = build_ensemble_predictions(predictions)
    tenor_specs = [
        {
            "label": "3年",
            "credit_candidates": ["中债中短期票据到期收益率(AAA):3年"],
            "treasury_candidates": ["中债国债到期收益率:3年"],
            "note": "AAA中短期票据3年",
        },
        {
            "label": "5年",
            "credit_candidates": ["中债中短期票据到期收益率(AAA):5年"],
            "treasury_candidates": ["中债国债到期收益率:5年"],
            "note": "AAA中短期票据5年",
        },
        {
            "label": "10年",
            "credit_candidates": ["中债中短期票据到期收益率(AAA):10年"],
            "treasury_candidates": ["中债国债到期收益率:10年"],
            "note": "AAA中短期票据10年",
        },
        {
            "label": "20年",
            "credit_candidates": [
                "中债中短期票据到期收益率(AAA):20年",
                "中债中短期票据到期收益率(AAA+):20年",
            ],
            "treasury_candidates": ["中债国债到期收益率:20年"],
            "note": "DM收益率曲线函数中AAA曲线暂不返回20年，当前20年继续使用AAA+口径",
        },
    ]
    tenors = []
    for spec in tenor_specs:
        credit_col = _find_column(columns, spec["credit_candidates"])
        treasury_col = _find_column(columns, spec["treasury_candidates"])
        points = _series_points(features, credit_col) if credit_col else []
        treasury_points = _series_points(features, treasury_col) if treasury_col else []
        spread_points = []
        if points and treasury_points:
            spread_df = features[["date", credit_col, treasury_col]].dropna().tail(20).copy()
            spread_df["_spread"] = spread_df[credit_col] - spread_df[treasury_col]
            spread_points = [
                {"date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"), "value": float(row["_spread"] * 100.0)}
                for _, row in spread_df.iterrows()
            ]
        model_prediction = prediction_by_target.get(credit_col or "")
        ensemble_prediction = ensemble_by_target.get(credit_col or "")
        tenors.append(
            {
                "label": spec["label"],
                "credit_col": credit_col or "",
                "treasury_col": treasury_col or "",
                "note": spec["note"],
                "available": bool(points),
                "latest_date": points[-1]["date"] if points else "",
                "latest_yield": _latest_value(points),
                "change_1d_bp": _bp_change(points, 1),
                "change_5d_bp": _bp_change(points, 5),
                "change_20d_bp": _bp_change(points, 19),
                "yield_points": points,
                "spread_latest_bp": _latest_value(spread_points),
                "spread_change_20d_bp": _raw_change(spread_points, 19),
                "spread_points": spread_points,
                "model_prediction": model_prediction,
                "ensemble_prediction": ensemble_prediction,
            }
        )
    return {
        "chart_start": pd.to_datetime(features["date"]).tail(20).min().strftime("%Y-%m-%d") if not features.empty else "",
        "chart_end": pd.to_datetime(features["date"]).tail(20).max().strftime("%Y-%m-%d") if not features.empty else "",
        "tenors": tenors,
    }


def _tenor_ensemble_items(report: dict) -> list[dict]:
    tenors = (report.get("market_snapshot") or {}).get("tenors", []) or []
    items = []
    for tenor in tenors:
        ensemble = tenor.get("ensemble_prediction") or {}
        probabilities = ensemble.get("probabilities") or {}
        if not probabilities:
            continue
        threshold_prediction = str(ensemble.get("threshold_prediction") or ensemble.get("raw_prediction") or "")
        raw_prediction = str(ensemble.get("direction_prediction") or ensemble.get("raw_prediction") or threshold_prediction)
        display_prediction = str(ensemble.get("display_prediction") or threshold_prediction)
        display_direction = str(ensemble.get("display_direction") or threshold_prediction)
        items.append(
            {
                "label": str(tenor.get("label") or tenor.get("credit_col") or "未命名期限"),
                "threshold_prediction": threshold_prediction,
                "raw_prediction": raw_prediction,
                "display_prediction": display_prediction,
                "display_direction": display_direction,
                "display_strength": str(ensemble.get("display_strength") or ""),
                "is_active": bool(ensemble.get("is_active")),
                "probabilities": probabilities,
            }
        )
    return items


def _average_probability_dicts(probability_dicts: list[dict[str, float]]) -> dict[str, float]:
    labels = ["看空", "看多", "震荡"]
    if not probability_dicts:
        return {label: 0.0 for label in labels}
    return {
        label: float(np.mean([float(item.get(label, 0.0)) for item in probability_dicts]))
        for label in labels
    }


def _term_list(items: list[dict]) -> str:
    labels = [item["label"] for item in items]
    return "、".join(labels) if labels else "无"


def _signal_summary(report: dict) -> dict:
    predictions = _valid_predictions(report)
    if not predictions:
        return {
            "title": "今日只完成数据更新",
            "direction": "暂无预测",
            "tone": "neutral",
            "strength": "无",
            "plain": "本次没有可展示的模型预测结果，可以先检查模型目录是否配置正确。",
            "avg_probabilities": _average_probabilities([]),
        }

    tenor_items = _tenor_ensemble_items(report)
    if tenor_items:
        avg = _average_probability_dicts([item["probabilities"] for item in tenor_items])
        display_counts = Counter(item["display_direction"] for item in tenor_items)
        display_bearish = [item for item in tenor_items if item["display_direction"] == "看空"]
        display_bullish = [item for item in tenor_items if item["display_direction"] == "看多"]
        strict_active_count = sum(1 for item in tenor_items if item["is_active"])
        display_active_count = len(display_bearish) + len(display_bullish)
        strong_count = sum(1 for item in tenor_items if item["display_strength"] == "强信号")

        if display_bearish and display_bullish:
            direction = "期限分化"
        elif display_counts.get("看空", 0) > 0:
            direction = "看空"
        elif display_counts.get("看多", 0) > 0:
            direction = "看多"
        elif display_counts.get("震荡", 0) > 0:
            direction = "震荡"
        else:
            direction = max(avg, key=avg.get)

        if direction == "看空":
            tone = "bearish"
            title = "期限信号汇总，综合偏看空"
            plain = "四个期限综合看，模型更担心信用债收益率上行，债券价格可能承压，因此不宜把它理解成追多信号。"
        elif direction == "看多":
            tone = "bullish"
            title = "期限信号汇总，综合偏看多"
            plain = "四个期限综合看，模型更倾向信用债收益率下行，这通常有利于债券价格表现，但仍需要结合资金面和信用事件复核。"
        elif direction == "期限分化":
            tone = "range"
            title = "期限方向分化，暂无单边综合信号"
            plain = "四个期限没有形成统一方向，当前更适合按期限结构分化处理，而不是把它理解成单边看多或单边看空。"
        else:
            tone = "range"
            if display_bearish and display_bullish:
                title = "综合震荡，期限方向分化"
            else:
                title = "综合震荡，方向信号偏弱"
            plain = "四个期限综合看，40/60展示口径下方向信号不足，当前更适合作为震荡或观望处理。"

        display_parts = []
        if display_bearish:
            display_parts.append(f"{_term_list(display_bearish)}震荡偏空")
        if display_bullish:
            display_parts.append(f"{_term_list(display_bullish)}震荡偏多")
        if display_parts:
            plain += " 40/60展示口径下，" + "，".join(display_parts) + "。"
        if strict_active_count < display_active_count:
            plain += " 其中部分倾向信号未通过更严格的45%与震荡优势过滤，因此仍应区分“偏向”和“强信号”。"

        top_prob = max(avg.values()) if avg else 0.0
        if direction == "期限分化":
            strength = "分化"
        elif strong_count > 0:
            strength = "较强"
        elif display_active_count > 0:
            strength = "倾向"
        elif strict_active_count == 0:
            strength = "偏弱"
        elif top_prob >= 0.60:
            strength = "较强"
        elif top_prob >= 0.50:
            strength = "中等"
        else:
            strength = "偏弱"

        return {
            "title": title,
            "direction": direction,
            "tone": tone,
            "strength": strength,
            "plain": plain,
            "avg_probabilities": avg,
        }

    labels = [item.get("prediction", "") for item in predictions]
    avg = _average_probabilities(predictions)
    direction = max(avg, key=avg.get)
    top_prob = avg[direction]
    unanimous = len(set(labels)) == 1
    if unanimous:
        title = f"模型一致：{direction}"
    else:
        title = f"模型分歧，综合偏{direction}"

    if not unanimous:
        strength = "偏弱"
    elif top_prob >= 0.70:
        strength = "较强"
    elif top_prob >= 0.55:
        strength = "中等"
    else:
        strength = "偏弱"

    if direction == "看空":
        plain = "模型更担心未来几个交易日信用债收益率上行。收益率上行通常意味着债券价格承压，因此这不是追多信号。"
        tone = "bearish"
    elif direction == "看多":
        plain = "模型更倾向未来几个交易日信用债收益率下行。收益率下行通常有利于债券价格表现，但仍需要结合资金面和信用事件复核。"
        tone = "bullish"
    else:
        plain = "模型更倾向市场维持震荡，方向信号不强。此时更适合维持纪律、观察新增数据，而不是主动加大方向暴露。"
        tone = "range"
    if not unanimous:
        plain += " 不同模型没有完全一致，所以应把它看成谨慎提示，而不是强信号。"

    return {
        "title": title,
        "direction": direction,
        "tone": tone,
        "strength": strength,
        "plain": plain,
        "avg_probabilities": avg,
    }


def _divergence_text(report: dict) -> str:
    predictions = _valid_predictions(report)
    if len(predictions) < 2:
        return "当前可用模型不足两个，暂不能判断模型分歧。"
    labels = [item.get("prediction", "") for item in predictions]
    if len(set(labels)) == 1:
        return f"当前 {len(predictions)} 个模型方向一致，均为“{labels[0]}”。"
    return "当前模型方向不完全一致，应降低信号置信度，并结合资金面、供给和信用事件复核。"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _probability_rows(probabilities: dict[str, float]) -> str:
    color_class = {"看空": "bar-bearish", "看多": "bar-bullish", "震荡": "bar-range"}
    rows = []
    for label in ["看空", "看多", "震荡"]:
        value = float(probabilities.get(label, 0.0))
        rows.append(
            f"""
            <div class="prob-row">
              <div class="prob-label">{html.escape(label)}</div>
              <div class="prob-track"><div class="prob-fill {color_class[label]}" style="width: {_pct(value)}"></div></div>
              <div class="prob-value">{_pct(value)}</div>
            </div>
            """
        )
    return "\n".join(rows)


def _fmt_bp(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "暂无"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f} bp"


def _fmt_pct_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "暂无"
    return f"{value:.4f}%"


def _fmt_prob(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "暂无"
    return _pct(float(value))


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "暂无"
    return f"{float(value):.{digits}f}"


def load_market_context_summary(market_context_dir: str | Path | None) -> dict:
    if not market_context_dir:
        return {}
    base = Path(market_context_dir)
    path = base if base.is_file() else base / "features" / "market_context_features.json"
    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def _market_context_block(report: dict) -> str:
    context = report.get("market_context") or {}
    if not context:
        return ""
    if context.get("missing") or context.get("error"):
        message = context.get("error") or f"未找到市场背景文件：{context.get('path', '')}"
        return f"""
        <section class="section">
          <h2>市场背景</h2>
          <p class="muted">{html.escape(str(message))}</p>
        </section>
        """
    tone = str(context.get("tone", "range"))
    display_signal = str(context.get("display_signal", "震荡"))
    score = context.get("pressure_score")
    intraday = context.get("intraday") or {}
    quote = context.get("credit_quote") or {}
    supply = context.get("primary_supply") or {}
    sentiment = context.get("money_sentiment") or {}
    dr007 = next(
        (item for item in context.get("money_market", []) if str(item.get("instrument_code", "")).upper() == "DR007"),
        {},
    )
    pressure_rows = []
    for item in context.get("pressure_parts", []):
        pressure_rows.append(
            f"""
            <div class="history-card">
              <span>{html.escape(str(item.get("name", "")))}</span>
              <strong>{_fmt_float(float(item.get("score", 0.0)), 1)}</strong>
            </div>
            """
        )
    diagnostics = "".join(f"<li>{html.escape(str(item))}</li>" for item in context.get("diagnostics", [])[:6])
    return f"""
    <section class="section">
      <h2>市场背景：资金、供给、报价与盘中压力</h2>
      <div class="context-head">
        <div>
          <span>日内市场信号</span>
          <strong class="{html.escape(tone)}">{html.escape(display_signal)}</strong>
        </div>
        <div>
          <span>市场压力分</span>
          <strong>{_fmt_float(float(score), 1)}</strong>
        </div>
        <div>
          <span>DR007 历史分位</span>
          <strong>{_fmt_prob(dr007.get("percentile"))}</strong>
        </div>
        <div>
          <span>信用债报价样本</span>
          <strong>{html.escape(str(quote.get("quote_count", 0)))} 条</strong>
        </div>
      </div>
      <p class="muted">{html.escape(str(context.get("plain", "")))}</p>
      <div class="meta-grid">
        <div class="metric"><span>国债活跃券日内</span><strong>{_fmt_bp(intraday.get("tbond_avg_yield_change_bp"))}</strong></div>
        <div class="metric"><span>国债期货日内</span><strong>{_fmt_prob(intraday.get("future_avg_price_change_pct"))}</strong></div>
        <div class="metric"><span>资金情绪分位</span><strong>{_fmt_prob(sentiment.get("percentile"))}</strong></div>
        <div class="metric"><span>一级发行规模分位</span><strong>{_fmt_prob(supply.get("amount_percentile"))}</strong></div>
      </div>
      <div class="history-grid" style="margin-top: 14px;">
        {"".join(pressure_rows)}
      </div>
      <ul class="context-list">{diagnostics}</ul>
    </section>
    """


def _tone_from_prediction(prediction: str) -> str:
    text = str(prediction)
    if "偏空" in text:
        return "tilt-bearish"
    if "偏多" in text:
        return "tilt-bullish"
    if "看空" in text:
        return "bearish"
    if "看多" in text:
        return "bullish"
    if "震荡" in text:
        return "range"
    return "neutral"


def _threshold_note(ensemble: dict) -> str:
    if not ensemble:
        return ""
    direction = str(ensemble.get("direction_prediction", ""))
    direction_prob = _fmt_prob(ensemble.get("direction_prob"))
    range_prob = _fmt_prob(ensemble.get("range_prob"))
    margin = float(ensemble.get("margin_vs_range", 0.0)) * 100.0
    prob_threshold = float(ensemble.get("prob_threshold", DAILY_SIGNAL_PROB_THRESHOLD)) * 100.0
    margin_threshold = float(ensemble.get("margin_threshold", DAILY_SIGNAL_MARGIN_VS_RANGE)) * 100.0
    status = str(ensemble.get("status", ""))
    if ensemble.get("is_active"):
        return (
            f"{status}：{direction}概率 {direction_prob}，震荡概率 {range_prob}，"
            f"相对震荡优势 {margin:+.1f} 个百分点。"
        )
    return (
        f"{status}：方向候选为{direction}，概率 {direction_prob}，震荡概率 {range_prob}，"
        f"相对震荡优势 {margin:+.1f} 个百分点；未同时满足概率≥{prob_threshold:.0f}%、"
        f"优势≥{margin_threshold:.0f}个百分点。"
    )


def _target_display_parts(target_col: str) -> dict[str, str]:
    text = str(target_col or "")
    tenor_match = re.search(r"(\d+)\s*年", text)
    tenor = f"{tenor_match.group(1)}年" if tenor_match else "未知期限"
    rating_match = re.search(r"\((AAA\+?|AA\+?|A\+?|AAA-?|AA-?)\)", text)
    rating = rating_match.group(1) if rating_match else ("AAA+" if "AAA+" in text else "AAA" if "AAA" in text else "信用债")
    return {"tenor": tenor, "rating": rating}


def _architecture_label(model_dir: str | Path) -> str:
    text = str(model_dir).lower()
    if re.search(r"(^|[\\/])gru($|[\\/])", text):
        return "GRU 循环神经网络"
    if re.search(r"(^|[\\/])tcn($|[\\/])", text):
        return "TCN 时序卷积网络"
    if re.search(r"(^|[\\/])transformer($|[\\/])", text):
        return "Transformer 注意力模型"
    if "lstm" in text:
        return "LSTM 循环神经网络"
    return "神经网络模型"


def _feature_set_label(model_dir: str | Path) -> str:
    text = str(model_dir).lower()
    if "01_full_features" in text:
        return "全特征"
    if "04_selected_features" in text:
        return "精选特征"
    if "07_positive_features" in text:
        return "正贡献特征"
    return "当前特征"


def _daily_model_name(item: dict, index: int) -> str:
    result = item.get("result", {}) if item.get("ok") else {}
    target = str(result.get("target_col", ""))
    parts = _target_display_parts(target)
    architecture = _architecture_label(item.get("model_dir", ""))
    feature_set = _feature_set_label(item.get("model_dir", ""))
    if target:
        return f"{parts['tenor']} {parts['rating']} - {architecture}（{feature_set}）"
    return f"模型 {index} - {_architecture_label(item.get('model_dir', ''))}"


def _daily_model_meta(item: dict) -> str:
    result = item.get("result", {}) if item.get("ok") else {}
    target = str(result.get("target_col", ""))
    model_dir = str(item.get("model_dir", ""))
    feature_set = _feature_set_label(model_dir)
    architecture = _architecture_label(model_dir)
    pieces = [f"结构：{architecture}", f"特征组：{feature_set}"]
    if target:
        pieces.append(f"目标：{target}")
    pieces.append(f"目录：{model_dir}")
    return "；".join(pieces)


def _sparkline(points: list[dict], color: str = "#2f6f9f") -> str:
    if len(points) < 2:
        return '<div class="spark-empty">暂无足够数据</div>'
    values = np.asarray([float(item["value"]) for item in points], dtype=float)
    min_v = float(np.nanmin(values))
    max_v = float(np.nanmax(values))
    raw_span = max(max_v - min_v, 1e-9)
    pad_value = max(raw_span * 0.18, 0.006)
    y_min = min_v - pad_value
    y_max = max_v + pad_value
    span = max(y_max - y_min, 1e-9)
    width = 300
    height = 128
    pad_l = 48
    pad_r = 10
    pad_t = 20
    pad_b = 26

    def x_pos(i: int) -> float:
        return pad_l + i * (width - pad_l - pad_r) / max(len(values) - 1, 1)

    def y_pos(value: float) -> float:
        return pad_t + (y_max - value) * (height - pad_t - pad_b) / span

    coords = []
    for i, value in enumerate(values):
        coords.append(f"{x_pos(i):.1f},{y_pos(float(value)):.1f}")

    ticks = [y_max, (y_min + y_max) / 2, y_min]
    grid = []
    for tick in ticks:
        y = y_pos(tick)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e3e9ee" stroke-width="1" />')
        grid.append(
            f'<text x="{pad_l - 7}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" fill="#607286">{tick:.2f}%</text>'
        )
    area_points = f"{pad_l:.1f},{height - pad_b:.1f} {' '.join(coords)} {width - pad_r:.1f},{height - pad_b:.1f}"
    last_x = x_pos(len(values) - 1)
    last_y = y_pos(float(values[-1]))
    first_date = html.escape(str(points[0]["date"]))
    last_date = html.escape(str(points[-1]["date"]))
    return f"""
    <div class="spark-wrap">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="近20日走势">
        <text x="{pad_l}" y="12" font-size="10.5" fill="#607286">收益率(%)</text>
        {''.join(grid)}
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#cfd8df" stroke-width="1" />
        <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#cfd8df" stroke-width="1" />
        <polygon points="{area_points}" fill="{color}" opacity="0.08" />
        <polyline points="{' '.join(coords)}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />
        <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.2" fill="#ffffff" stroke="{color}" stroke-width="2" />
      </svg>
      <div class="spark-dates"><span>{first_date}</span><span>{last_date}</span></div>
    </div>
    """


def _term_blocks(report: dict) -> str:
    market = report.get("market_snapshot", {})
    summary = _signal_summary(report)
    blocks = []
    for tenor in market.get("tenors", []):
        label = html.escape(str(tenor.get("label", "")))
        if not tenor.get("available"):
            blocks.append(
                f"""
                <section class="term-card">
                  <div class="term-head"><h3>{label}</h3><span class="pill neutral">暂无数据</span></div>
                  <p class="muted">DM当前配置没有可用收益率数据。</p>
                </section>
                """
            )
            continue
        ensemble = tenor.get("ensemble_prediction") or {}
        model_prediction = tenor.get("model_prediction")
        if ensemble:
            pred_text = str(ensemble.get("display_prediction") or ensemble.get("threshold_prediction", ""))
            strict_text = str(ensemble.get("threshold_prediction", ""))
            direction_text = str(ensemble.get("direction_prediction", ""))
            tone = _tone_from_prediction(pred_text)
            model_count = int(ensemble.get("model_count", 0))
            pred_note = (
                f"40/60展示信号：{pred_text}（{model_count}个模型平均）；"
                f"严格阈值后信号：{strict_text}；方向候选：{direction_text}。"
                f"{_threshold_note(ensemble)}"
            )
            probability_block = f"""
              <div class="ensemble-probs">
                <div class="signal-subtitle">集成概率</div>
                {_probability_rows(ensemble.get("probabilities", {}))}
              </div>
            """
        elif model_prediction:
            raw_probs = _prob_vector(model_prediction)
            threshold = _threshold_signal_from_probs(raw_probs)
            pred_text = str(threshold.get("display_prediction", ""))
            strict_text = str(threshold.get("threshold_prediction", ""))
            direction_text = str(threshold.get("direction_prediction", ""))
            tone = _tone_from_prediction(pred_text)
            pred_note = (
                f"40/60展示信号：{pred_text}；"
                f"严格阈值后信号：{strict_text}；方向候选：{direction_text}。"
                f"{_threshold_note(threshold)}"
            )
            probability_block = f"""
              <div class="ensemble-probs">
                <div class="signal-subtitle">模型概率</div>
                {_probability_rows(_prob_dict_from_vector(raw_probs))}
              </div>
            """
        else:
            pred_text = str(summary.get("direction", ""))
            tone = _tone_from_prediction(pred_text)
            pred_note = f"总方向参考：{html.escape(pred_text)}；期限专属模型待训练"
            probability_block = ""
        blocks.append(
            f"""
            <section class="term-card">
              <div class="term-head">
                <h3>{label}</h3>
                <span class="pill {tone}">{html.escape(pred_text or "观察")}</span>
              </div>
              <p class="muted date-line">数据日期：{html.escape(str(tenor.get("latest_date", "")))}</p>
              {_sparkline(tenor.get("yield_points", []), "#2f6f9f")}
              <div class="term-metrics">
                <div><span>最新收益率</span><strong>{_fmt_pct_value(tenor.get("latest_yield"))}</strong></div>
                <div><span>1日变化</span><strong>{_fmt_bp(tenor.get("change_1d_bp"))}</strong></div>
                <div><span>5日变化</span><strong>{_fmt_bp(tenor.get("change_5d_bp"))}</strong></div>
                <div><span>20日变化</span><strong>{_fmt_bp(tenor.get("change_20d_bp"))}</strong></div>
              </div>
              <div class="spread-line">
                信用利差：{_fmt_bp(tenor.get("spread_latest_bp"))}；
                近20日变化：{_fmt_bp(tenor.get("spread_change_20d_bp"))}
              </div>
              {probability_block}
              <p class="muted">{html.escape(pred_note)} {html.escape(str(tenor.get("note", "")))}</p>
            </section>
            """
        )
    return "\n".join(blocks) if blocks else '<p class="muted">暂无期限走势数据。</p>'


def _history_blocks(report: dict) -> str:
    blocks = []
    for index, item in enumerate(report.get("predictions", []), start=1):
        if not item.get("ok"):
            continue
        stats = item.get("history_stats", {})
        model_name = _daily_model_name(item, index)
        model_meta = _daily_model_meta(item)
        if not stats.get("has_history"):
            blocks.append(
                f"""
                <div class="history-card">
                  <strong>{html.escape(model_name)}</strong>
                  <span>{html.escape(model_meta)}</span>
                  <span>暂无历史测试记录</span>
                </div>
                """
            )
            continue
        accuracy = stats.get("accuracy")
        macro_f1 = stats.get("macro_f1")
        active = stats.get("active_signal_ratio")
        ret = stats.get("total_return_proxy")
        blocks.append(
            f"""
            <div class="history-card">
              <strong>{html.escape(model_name)}</strong>
              <span>{html.escape(model_meta)}</span>
              <span>测试区间：{html.escape(str(stats.get("test_start", "")))} 至 {html.escape(str(stats.get("test_end", "")))}</span>
              <span>准确率：{_fmt_prob(accuracy)}；宏平均F1：{_fmt_float(macro_f1, 3)}</span>
              <span>收益代理：{_fmt_float(ret, 4)}；活跃信号占比：{_fmt_prob(active)}</span>
            </div>
            """
        )
    return "\n".join(blocks) if blocks else '<p class="muted">暂无历史表现数据。</p>'


def _model_prediction_blocks(report: dict) -> str:
    blocks = []
    for index, item in enumerate(report.get("predictions", []), start=1):
        model_name = _daily_model_name(item, index)
        model_meta = _daily_model_meta(item)
        if not item.get("ok"):
            blocks.append(
                f"""
                <section class="model-block">
                  <div class="model-head">
                    <h3>{html.escape(model_name)}</h3>
                    <span class="pill neutral">预测失败</span>
                  </div>
                  <p class="model-meta">{html.escape(model_meta)}</p>
                  <p class="muted">{html.escape(str(item.get("error", "")))}</p>
                </section>
                """
            )
            continue
        result = item["result"]
        prediction = str(result.get("prediction", ""))
        tone = {"看空": "bearish", "看多": "bullish", "震荡": "range"}.get(prediction, "neutral")
        blocks.append(
            f"""
            <section class="model-block">
              <div class="model-head">
                <h3>{html.escape(model_name)}</h3>
                <span class="pill {tone}">{html.escape(prediction)}</span>
              </div>
              <p class="model-meta">{html.escape(model_meta)}</p>
              <div class="prob-list">
                {_probability_rows(result.get("probabilities", {}))}
              </div>
              <p class="muted">{html.escape(str(result.get("advice", "")))}</p>
            </section>
            """
        )
    if not blocks:
        return '<p class="muted">本次没有配置模型预测。</p>'
    return "\n".join(blocks)


def write_daily_html(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _signal_summary(report)
    avg = summary["avg_probabilities"]
    prediction_date = ""
    predictions = _valid_predictions(report)
    if predictions:
        prediction_date = str(predictions[0].get("prediction_date", ""))
        horizon = str(predictions[0].get("horizon_days", ""))
    else:
        horizon = ""

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债每日 AI 观察</title>
  <style>
    :root {{
      --ink: #141b2d;
      --muted: #64748b;
      --line: #d8e1ea;
      --paper: #ffffff;
      --panel: #f8fafc;
      --back: #f3f6fa;
      --bearish: #16865a;
      --bullish: #d64545;
      --range: #b98528;
      --blue: #246f9f;
      --blue-soft: #eaf3f8;
      --shadow: 0 10px 28px rgba(20, 36, 56, 0.07);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--back);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    .top {{
      background: var(--paper);
      border-bottom: 1px solid var(--line);
      border-top: 4px solid var(--blue);
    }}
    .wrap {{
      width: min(1180px, calc(100% - 36px));
      margin: 0 auto;
    }}
    .hero {{
      padding: 24px 0 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 22px;
      align-items: stretch;
    }}
    .eyebrow {{
      color: var(--blue);
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .hero p {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 820px;
      font-size: 15px;
    }}
    .signal-box {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      border-radius: 8px;
      display: grid;
      align-content: center;
      box-shadow: 0 8px 22px rgba(20, 36, 56, 0.05);
    }}
    .signal-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .signal-value {{
      font-size: 30px;
      font-weight: 800;
      line-height: 1.15;
    }}
    .signal-value.bearish {{ color: var(--bearish); }}
    .signal-value.bullish {{ color: var(--bullish); }}
    .signal-value.tilt-bearish {{ color: var(--bearish); }}
    .signal-value.tilt-bullish {{ color: var(--bullish); }}
    .signal-value.range {{ color: var(--range); }}
    .signal-value.neutral {{ color: var(--muted); }}
    .main {{
      padding: 18px 0 34px;
      display: grid;
      gap: 14px;
    }}
    .section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .takeaway {{
      font-size: 15.5px;
      margin: 0;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border-top: 2px solid var(--blue);
      background: var(--panel);
      padding: 10px 11px;
      border-radius: 6px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 16px;
    }}
    .model-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .term-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .model-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfdff;
      box-shadow: 0 5px 16px rgba(20, 36, 56, 0.045);
    }}
    .term-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfdff;
      box-shadow: 0 5px 16px rgba(20, 36, 56, 0.045);
    }}
    .model-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .term-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .model-head h3 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.35;
    }}
    .term-head h3 {{
      margin: 0;
      font-size: 20px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 58px;
      padding: 5px 10px;
      border-radius: 999px;
      color: #fff;
      font-weight: 700;
      font-size: 13px;
    }}
    .pill.bearish {{ background: var(--bearish); }}
    .pill.bullish {{ background: var(--bullish); }}
    .pill.tilt-bearish {{ background: #e4f5ed; color: #126b47; border: 1px solid #b9e4cf; }}
    .pill.tilt-bullish {{ background: #fde7e7; color: #ad3232; border: 1px solid #f3c0c0; }}
    .pill.range {{ background: var(--range); }}
    .pill.neutral {{ background: var(--muted); }}
    .prob-list {{
      display: grid;
      gap: 10px;
    }}
    .prob-row {{
      display: grid;
      grid-template-columns: 44px 1fr 54px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
    }}
    .prob-track {{
      height: 10px;
      background: #e7edf3;
      border-radius: 999px;
      overflow: hidden;
    }}
    .prob-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-bearish {{ background: var(--bearish); }}
    .bar-bullish {{ background: var(--bullish); }}
    .bar-range {{ background: var(--range); }}
    .prob-value {{
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .muted {{
      color: var(--muted);
      margin: 12px 0 0;
      font-size: 14px;
    }}
    .model-meta {{
      color: var(--muted);
      margin: -4px 0 14px;
      font-size: 12px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
    .date-line {{
      margin-top: 0;
    }}
    .spark-wrap {{
      margin: 10px 0;
    }}
    .spark-wrap svg {{
      width: 100%;
      height: 116px;
      display: block;
      background: #fff;
      border: 1px solid #e7edf3;
      border-radius: 6px;
    }}
    .spark-dates {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }}
    .spark-empty {{
      color: var(--muted);
      background: #fff;
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 22px;
      text-align: center;
    }}
    .term-metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .term-metrics div {{
      background: #fff;
      border: 1px solid #e7edf3;
      border-radius: 6px;
      padding: 9px;
    }}
    .term-metrics span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .term-metrics strong {{
      display: block;
      margin-top: 2px;
      font-size: 14px;
    }}
    .spread-line {{
      margin-top: 10px;
      padding: 9px 10px;
      border-radius: 6px;
      background: var(--blue-soft);
      color: #255873;
      font-size: 13px;
    }}
    .ensemble-probs {{
      margin-top: 10px;
      padding: 10px;
      border: 1px solid #e7edf3;
      border-radius: 6px;
      background: #ffffff;
    }}
    .signal-subtitle {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .history-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .history-card {{
      display: grid;
      gap: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfdff;
      color: var(--muted);
      font-size: 13px;
      box-shadow: 0 5px 16px rgba(20, 36, 56, 0.04);
    }}
    .history-card strong {{
      color: var(--ink);
      font-size: 16px;
    }}
    .context-head {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .context-head div {{
      background: var(--panel);
      border: 1px solid #e7edf3;
      border-radius: 8px;
      padding: 12px;
    }}
    .context-head span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .context-head strong {{
      display: block;
      margin-top: 5px;
      font-size: 18px;
    }}
    .context-head strong.bearish {{ color: var(--bearish); }}
    .context-head strong.bullish {{ color: var(--bullish); }}
    .context-head strong.range {{ color: var(--range); }}
    .context-list {{
      margin: 12px 0 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.8;
    }}
    .note {{
      border-left: 4px solid var(--range);
      padding: 12px 14px;
      background: #fff7e8;
      border-radius: 6px;
      color: #493622;
    }}
    .file-list {{
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 780px) {{
      .hero, .model-grid, .meta-grid, .term-grid, .history-grid, .context-head {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 28px;
      }}
      .section {{
        padding: 18px;
      }}
      .term-metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <header class="top">
    <div class="wrap hero">
      <div>
        <div class="eyebrow">信用债每日 AI 观察</div>
        <h1>{html.escape(summary["title"])}</h1>
        <p>{html.escape(summary["plain"])}</p>
      </div>
      <div class="signal-box">
        <div class="signal-label">综合方向</div>
        <div class="signal-value {summary["tone"]}">{html.escape(summary["direction"])}</div>
        <div class="muted">信号强度：{html.escape(summary["strength"])}</div>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{html.escape(summary["plain"])}</p>
      <div class="meta-grid">
        <div class="metric"><span>预测日期</span><strong>{html.escape(prediction_date or report.get("data_end", ""))}</strong></div>
        <div class="metric"><span>预测周期</span><strong>{html.escape(horizon or "未配置")} 个交易日</strong></div>
        <div class="metric"><span>数据范围</span><strong>{html.escape(report.get("data_start", ""))} 至 {html.escape(report.get("data_end", ""))}</strong></div>
        <div class="metric"><span>特征数量</span><strong>{html.escape(str(report.get("features_shape", ["", ""])[1]))}</strong></div>
      </div>
    </section>

    {_market_context_block(report)}

    <section class="section">
      <h2>综合概率</h2>
      <div class="prob-list">
        {_probability_rows(avg)}
      </div>
    </section>

    <section class="section">
      <h2>3年 / 5年 / 10年 / 20年期限走势</h2>
      <div class="term-grid">
        {_term_blocks(report)}
      </div>
    </section>

    <section class="section">
      <h2>模型明细与参照</h2>
      <div class="model-grid">
        {_model_prediction_blocks(report)}
      </div>
    </section>

    <section class="section">
      <h2>模型分歧与历史表现</h2>
      <div class="note">{html.escape(_divergence_text(report))}</div>
      <div class="history-grid" style="margin-top: 14px;">
        {_history_blocks(report)}
      </div>
    </section>

    <section class="section">
      <h2>怎么读</h2>
      <div class="note">这是一份研究辅助输出，不是自动交易指令。看空表示模型更担心收益率上行、债券价格承压；看多表示模型更倾向收益率下行；震荡表示方向优势不明显。</div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_daily_markdown(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = report.get("predictions", [])
    pred_lines = []
    for item in predictions:
        if item.get("ok"):
            pred_lines.append(
                f"- `{item['model_dir']}`：{_summarize_prediction(item['result'])}"
            )
        else:
            pred_lines.append(f"- `{item.get('model_dir', '')}`：失败，{item.get('error', '')}")
    if not pred_lines:
        pred_lines.append("- 未配置模型目录，本次只完成 DM 数据更新。")

    content = f"""# 每日 DM 更新报告

## 基本信息

- 运行日期：{report.get('run_date', '')}
- 配置文件：`{report.get('config_path', '')}`
- 原始数据目录：`{report.get('raw_dir', '')}`
- 特征文件：`{report.get('features_path', '')}`
- 数据日期范围：{report.get('data_start', '')} 至 {report.get('data_end', '')}
- 特征表形状：{report.get('features_shape', '')}

## 预测结果

{chr(10).join(pred_lines)}

## 文件

- 防穿越宽表：`{report.get('wide_path', '')}`
- 模型就绪表：`{report.get('model_ready_path', '')}`
- 数据字典：`{report.get('dictionary_path', '')}`
- HTML 报告：`{report.get('html_report_path', '')}`
- JSON 报告：`{report.get('json_report_path', '')}`
"""
    path.write_text(content, encoding="utf-8")


def run_daily_dm_update(
    config_path: str | Path,
    out_dir: str | Path = "data/dm_daily",
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    freq: str = "B",
    fallback_release_lag_days: int = 10,
    model_ready_max_missing_ratio: float = 0.2,
    derive_features: bool = True,
    skip_fetch: bool = False,
    incremental: bool = True,
    overlap_days: int = 15,
    model_dirs: Iterable[str] | None = None,
    models_root: str | Path | None = None,
    device_name: str = "auto",
    base_url: str | None = None,
    timeout: int = 30,
    strict_predictions: bool = False,
    market_context_dir: str | Path | None = None,
) -> dict:
    end_date = end_date or date.today().isoformat()
    paths = build_daily_paths(out_dir, raw_dir=raw_dir, processed_dir=processed_dir, report_dir=report_dir)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    config = load_dm_indicator_config(config_path)
    fetch_rows: list[dict] = []
    if not skip_fetch:
        fetch_rows = fetch_dm_raw_incremental(
            config=config,
            raw_dir=paths.raw_dir,
            default_start_date=start_date,
            end_date=end_date,
            incremental=incremental,
            overlap_days=overlap_days,
            base_url=base_url,
            timeout=timeout,
        )

    wide_report = edb_raw_to_point_in_time_wide(
        input_path=paths.raw_dir,
        output_path=paths.wide_path,
        dictionary_path=paths.dictionary_path,
        model_ready_path=paths.model_ready_path,
        freq=freq,
        start_date=start_date,
        end_date=end_date,
        fallback_release_lag_days=fallback_release_lag_days,
        model_ready_max_missing_ratio=model_ready_max_missing_ratio,
    )
    features = prepare_wide_data(
        input_path=paths.model_ready_path,
        output_path=paths.features_path,
        freq=freq,
        fill="ffill",
        derive_features=derive_features,
    )

    prediction_rows = []
    for model_dir in find_model_dirs(model_dirs, models_root=models_root):
        output_name = _safe_name(str(model_dir).replace("\\", "_").replace("/", "_"))
        prediction_path = paths.report_dir / f"prediction_{output_name}.json"
        try:
            result = predict_latest(
                model_dir=model_dir,
                data_path=paths.features_path,
                out_path=prediction_path,
                device_name=device_name,
            )
            prediction_rows.append(
                {
                    "ok": True,
                    "model_dir": str(model_dir),
                    "prediction_path": str(prediction_path),
                    "result": result,
                    "history_stats": _model_history_stats(model_dir),
                }
            )
        except Exception as exc:
            if strict_predictions:
                raise
            prediction_rows.append({"ok": False, "model_dir": str(model_dir), "error": str(exc), "history_stats": _model_history_stats(model_dir)})

    valid_prediction_results = [item["result"] for item in prediction_rows if item.get("ok")]
    market_snapshot = build_market_snapshot(features, valid_prediction_results)
    market_context = load_market_context_summary(market_context_dir)

    report = {
        "run_date": date.today().isoformat(),
        "config_path": str(config_path),
        "end_date": end_date,
        "raw_dir": str(paths.raw_dir),
        "processed_dir": str(paths.processed_dir),
        "report_dir": str(paths.report_dir),
        "wide_path": str(paths.wide_path),
        "dictionary_path": str(paths.dictionary_path),
        "model_ready_path": str(paths.model_ready_path),
        "features_path": str(paths.features_path),
        "json_report_path": str(paths.json_report_path),
        "markdown_report_path": str(paths.markdown_report_path),
        "html_report_path": str(paths.html_report_path),
        "fetch": fetch_rows,
        "wide_report": wide_report,
        "features_shape": list(features.shape),
        "data_start": pd.to_datetime(features["date"]).min().strftime("%Y-%m-%d") if not features.empty else "",
        "data_end": pd.to_datetime(features["date"]).max().strftime("%Y-%m-%d") if not features.empty else "",
        "predictions": prediction_rows,
        "market_snapshot": market_snapshot,
        "market_context": market_context,
    }
    with paths.json_report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    write_daily_markdown(report, paths.markdown_report_path)
    write_daily_html(report, paths.html_report_path)
    return report
