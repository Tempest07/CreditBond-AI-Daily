from __future__ import annotations

import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .data import add_derived_features, build_windowed_dataset, load_wide_dataset
from .model import ModelConfig
from .position_backtest import (
    _infer_duration,
    _load_checkpoint,
    _load_test_predictions,
    _prepare_asset_returns,
    _predict_all_history,
    parse_duration_map,
)
from .training import TrainConfig, train_model


LABEL_CN = {
    0: "看空",
    1: "看多",
    2: "震荡",
}


@dataclass
class StrategySweepConfig:
    features_path: str | Path
    model_dirs: list[str | Path]
    out_dir: str | Path
    prob_thresholds: list[float]
    margins: list[float]
    steps: list[float]
    signal_scope: str = "all"
    initial_position: float = 0.5
    transaction_cost_bp: float = 0.0
    annual_days: int = 252
    include_carry: bool = True
    include_ensembles: bool = True
    duration_map: dict[str, float] | None = None


@dataclass
class LabelSweepConfig:
    data_path: str | Path
    target_cols: list[str]
    out_dir: str | Path
    theta_quantiles: list[float]
    models: list[str]
    horizon: int = 5
    window: int = 60
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    train_end: str | None = None
    val_end: str | None = None
    exclude_target_feature: bool = True
    derive_features: bool = False
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 5
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.2
    heads: int = 4
    kernel_size: int = 3
    device: str = "auto"
    class_weight: bool = True
    duration_map: dict[str, float] | None = None
    yield_unit: str = "percent"
    seed: int = 42


@dataclass
class RollingValidationConfig:
    data_path: str | Path
    target_cols: list[str]
    out_dir: str | Path
    models: list[str]
    theta_quantile: float = 0.6
    horizon: int = 5
    window: int = 60
    min_train_rows: int = 180
    val_rows: int = 45
    test_rows: int = 45
    step_rows: int = 45
    max_folds: int = 3
    exclude_target_feature: bool = True
    derive_features: bool = False
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 4
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.2
    heads: int = 4
    kernel_size: int = 3
    device: str = "auto"
    class_weight: bool = True
    duration_map: dict[str, float] | None = None
    yield_unit: str = "percent"
    seed: int = 42


def parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if text is None or not str(text).strip():
        return default
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_str_list(text: str | None, default: list[str]) -> list[str]:
    if text is None or not str(text).strip():
        return default
    return [part.strip() for part in str(text).split(",") if part.strip()]


def decode_cli_text(text: str) -> str:
    if "\\u" not in str(text) and "\\U" not in str(text):
        return str(text)
    return str(text).encode("utf-8").decode("unicode_escape")


def decode_cli_text_list(values: list[str]) -> list[str]:
    return [decode_cli_text(value) for value in values]


def _jsonable_config(cfg: Any) -> dict[str, Any]:
    result = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def _safe_name(text: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(text), flags=re.UNICODE)
    return text.strip("_") or "item"


def _infer_tenor(target_col: str) -> str:
    match = re.search(r"(\d+)\s*年", str(target_col))
    return match.group(1) if match else ""


def _rating_text(target_col: str) -> str:
    return "AAA+" if "AAA+" in str(target_col) else "AAA"


def _format_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _format_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _annualized_return(total_return: float, periods: int, annual_days: int) -> float:
    if periods <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (annual_days / periods) - 1.0)


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series, annual_days: int) -> float | None:
    if len(returns) < 2:
        return None
    std = float(returns.std(ddof=1))
    if not np.isfinite(std) or std == 0:
        return None
    return float(returns.mean() / std * math.sqrt(annual_days))


def _clamp_position(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _read_features(path: str | Path) -> pd.DataFrame:
    features = load_wide_dataset(path)
    return features.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any").reset_index(drop=True)


def _model_label(model_dir: Path, target_col: str) -> str:
    tenor = _infer_tenor(target_col)
    rating = _rating_text(target_col)
    arch = model_dir.name
    feature_set = model_dir.parent.name
    tenor_text = f"{tenor}年" if tenor else ""
    return f"{rating}{tenor_text} {feature_set}/{arch}"


def _load_single_entity(
    model_dir: str | Path,
    features: pd.DataFrame,
    cfg: StrategySweepConfig,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    checkpoint = _load_checkpoint(model_dir)
    target_col = str(checkpoint.get("target_col", ""))
    if not target_col:
        raise ValueError(f"{model_dir} 缺少 target_col。")
    duration = _infer_duration(target_col, checkpoint, cfg.duration_map or {})
    if cfg.signal_scope == "all":
        pred = _predict_all_history(model_dir, checkpoint, features)
    elif cfg.signal_scope == "test":
        pred = _load_test_predictions(model_dir)
    else:
        raise ValueError("signal_scope must be all or test.")
    required = {"date", "prob_bearish", "prob_bullish", "prob_range"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{model_dir} 历史预测缺少字段：{sorted(missing)}")
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return {
        "entity_id": _safe_name(str(model_dir)),
        "entity_type": "single",
        "display_name": _model_label(model_dir, target_col),
        "target_col": target_col,
        "tenor": _infer_tenor(target_col),
        "rating": _rating_text(target_col),
        "duration": duration,
        "model_dirs": [str(model_dir)],
        "predictions": pred,
    }


def _make_ensemble_entities(single_entities: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    ensembles = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for entity in single_entities:
        groups.setdefault(entity["target_col"], []).append(entity)

    ens_dir = out_dir / "ensemble_predictions"
    ens_dir.mkdir(parents=True, exist_ok=True)
    for target_col, items in groups.items():
        if len(items) < 2:
            continue
        base = None
        prob_frames = []
        truth_frames = []
        for idx, item in enumerate(items):
            pred = item["predictions"].copy().set_index("date")
            prob_frames.append(pred[["prob_bearish", "prob_bullish", "prob_range"]].add_suffix(f"__m{idx}"))
            if "y_true" in pred.columns:
                truth_frames.append(pred[["y_true"]].rename(columns={"y_true": f"y_true__m{idx}"}))
            base = pred if base is None else base
        merged = pd.concat(prob_frames + truth_frames, axis=1, join="inner").dropna(how="any")
        if merged.empty:
            continue
        out = pd.DataFrame(index=merged.index)
        for col in ["prob_bearish", "prob_bullish", "prob_range"]:
            cols = [c for c in merged.columns if c.startswith(f"{col}__m")]
            out[col] = merged[cols].mean(axis=1)
        truth_cols = [c for c in merged.columns if c.startswith("y_true__m")]
        if truth_cols:
            out["y_true"] = merged[truth_cols[0]].astype(int)
        out = out.reset_index().sort_values("date")
        out["y_pred"] = out[["prob_bearish", "prob_bullish", "prob_range"]].to_numpy().argmax(axis=1)
        out["y_pred_cn"] = out["y_pred"].map(LABEL_CN)
        tenor = _infer_tenor(target_col)
        rating = _rating_text(target_col)
        entity_id = f"ensemble_{rating}_{tenor}Y"
        out.to_csv(ens_dir / f"{entity_id}.csv", index=False, encoding="utf-8-sig")
        ensembles.append(
            {
                "entity_id": entity_id,
                "entity_type": "ensemble",
                "display_name": f"{rating}{tenor}年 集成({len(items)}个模型平均)",
                "target_col": target_col,
                "tenor": tenor,
                "rating": rating,
                "duration": float(np.mean([item["duration"] for item in items])),
                "model_dirs": [path for item in items for path in item["model_dirs"]],
                "predictions": out,
            }
        )
    return ensembles


def _apply_probability_rule(pred: pd.DataFrame, min_action_prob: float, min_margin_vs_range: float) -> pd.DataFrame:
    out = pred.copy()
    bear = out["prob_bearish"].astype(float)
    bull = out["prob_bullish"].astype(float)
    range_prob = out["prob_range"].astype(float)
    bullish_side = bull >= bear
    action_prob = np.where(bullish_side, bull, bear)
    action_label = np.where(bullish_side, 1, 0)
    action_signal = np.where(bullish_side, 1, -1)
    active = (action_prob >= min_action_prob) & ((action_prob - range_prob) >= min_margin_vs_range)
    out["rule_prob_threshold"] = float(min_action_prob)
    out["rule_margin_vs_range"] = float(min_margin_vs_range)
    out["rule_action_prob"] = action_prob
    out["rule_signal"] = np.where(active, action_signal, 0).astype(int)
    out["rule_pred"] = np.where(active, action_label, 2).astype(int)
    out["rule_pred_cn"] = out["rule_pred"].map(LABEL_CN)
    return out


def _simulate_strategy(
    entity: dict[str, Any],
    pred: pd.DataFrame,
    features: pd.DataFrame,
    cfg: StrategySweepConfig,
    step: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    asset = _prepare_asset_returns(
        features=features,
        target_col=entity["target_col"],
        duration=float(entity["duration"]),
        annual_days=cfg.annual_days,
        include_carry=cfg.include_carry,
    )
    merged = pred.merge(asset, on="date", how="inner")
    if merged.empty:
        raise ValueError(f"{entity['display_name']} 预测日期和资产收益日期无法匹配。")

    position = _clamp_position(cfg.initial_position)
    strategy_value = 1.0
    buy_hold_value = 1.0
    neutral_value = 1.0
    rows = []
    for _, row in merged.iterrows():
        signal = int(row["rule_signal"])
        before = position
        if signal > 0:
            position = _clamp_position(position + step)
        elif signal < 0:
            position = _clamp_position(position - step)
        trade = abs(position - before)
        cost_return = trade * (cfg.transaction_cost_bp / 10000.0)
        bond_return = float(row["bond_return_proxy"])
        strategy_return = position * bond_return - cost_return
        buy_hold_return = bond_return
        neutral_return = cfg.initial_position * bond_return
        strategy_value *= 1.0 + strategy_return
        buy_hold_value *= 1.0 + buy_hold_return
        neutral_value *= 1.0 + neutral_return
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "next_date": pd.Timestamp(row["next_date"]).date().isoformat(),
                "rule_pred": int(row["rule_pred"]),
                "rule_pred_cn": LABEL_CN[int(row["rule_pred"])],
                "signal": signal,
                "position_before": before,
                "position_after": position,
                "trade": trade,
                "yield": float(row[entity["target_col"]]),
                "next_yield": float(row["next_yield"]),
                "yield_change_1d": float(row["yield_change_1d"]),
                "bond_return_proxy": bond_return,
                "strategy_return": strategy_return,
                "strategy_value": strategy_value,
                "buy_hold_return": buy_hold_return,
                "buy_hold_value": buy_hold_value,
                "neutral_return": neutral_return,
                "neutral_value": neutral_value,
                "rule_action_prob": float(row["rule_action_prob"]),
            }
        )
    detail = pd.DataFrame(rows)
    periods = len(detail)
    total_return = float(detail["strategy_value"].iloc[-1] - 1.0)
    buy_hold_total = float(detail["buy_hold_value"].iloc[-1] - 1.0)
    neutral_total = float(detail["neutral_value"].iloc[-1] - 1.0)
    active = detail["signal"].ne(0)
    metrics = {
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "display_name": entity["display_name"],
        "target_col": entity["target_col"],
        "tenor": entity["tenor"],
        "rating": entity["rating"],
        "model_count": len(entity["model_dirs"]),
        "model_dirs": "|".join(entity["model_dirs"]),
        "signal_scope": cfg.signal_scope,
        "prob_threshold": float(pred["rule_prob_threshold"].iloc[0]),
        "margin_vs_range": float(pred["rule_margin_vs_range"].iloc[0]),
        "position_step": float(step),
        "start_date": str(detail["date"].iloc[0]),
        "end_date": str(detail["next_date"].iloc[-1]),
        "periods": periods,
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, periods, cfg.annual_days),
        "max_drawdown": _max_drawdown(detail["strategy_value"]),
        "daily_volatility_annualized": float(detail["strategy_return"].std(ddof=1) * math.sqrt(cfg.annual_days)) if periods > 1 else 0.0,
        "sharpe": _sharpe(detail["strategy_return"], cfg.annual_days),
        "positive_day_ratio": float((detail["strategy_return"] > 0).mean()),
        "active_signal_ratio": float(active.mean()),
        "active_win_ratio": float((detail.loc[active, "strategy_return"] > 0).mean()) if active.any() else None,
        "average_action_prob": float(detail.loc[active, "rule_action_prob"].mean()) if active.any() else None,
        "final_position": float(detail["position_after"].iloc[-1]),
        "average_position": float(detail["position_after"].mean()),
        "turnover": float(detail["trade"].sum()),
        "trade_count": int((detail["trade"] > 0).sum()),
        "buy_hold_total_return": buy_hold_total,
        "buy_hold_annualized_return": _annualized_return(buy_hold_total, periods, cfg.annual_days),
        "neutral_total_return": neutral_total,
        "neutral_annualized_return": _annualized_return(neutral_total, periods, cfg.annual_days),
        "excess_vs_buy_hold": total_return - buy_hold_total,
        "excess_vs_neutral": total_return - neutral_total,
    }
    if "y_true" in merged.columns:
        y_true = merged["y_true"].astype(int).to_numpy()
        y_pred = merged["rule_pred"].astype(int).to_numpy()
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        action_mask = y_pred != 2
        metrics["active_label_hit_ratio"] = float((y_true[action_mask] == y_pred[action_mask]).mean()) if action_mask.any() else None
    else:
        metrics["accuracy"] = None
        metrics["macro_f1"] = None
        metrics["active_label_hit_ratio"] = None
    return metrics, detail


def _best_by_group(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if detail.empty:
        return detail.copy()
    sort_cols = group_cols + ["total_return", "excess_vs_buy_hold", "sharpe"]
    work = detail.sort_values(sort_cols, ascending=[True] * len(group_cols) + [False, False, False], na_position="last")
    return work.groupby(group_cols, dropna=False).head(1).reset_index(drop=True)


def _write_strategy_html(report: dict[str, Any], path: Path) -> None:
    detail = pd.DataFrame(report["detail"])
    best_entity = pd.DataFrame(report["best_by_entity"])
    top = detail.sort_values(["total_return", "excess_vs_buy_hold"], ascending=False).head(20)
    ensembles = detail[detail["entity_type"].eq("ensemble")].sort_values("total_return", ascending=False).head(12)
    best = top.iloc[0] if not top.empty else None
    baseline = detail[
        detail["prob_threshold"].round(6).eq(0.0)
        & detail["margin_vs_range"].round(6).eq(0.0)
        & detail["position_step"].round(6).eq(0.05)
    ]
    baseline_best = baseline.sort_values("total_return", ascending=False).head(1)

    def esc(value: Any) -> str:
        return html.escape(str(value))

    def metric_card(title: str, value: str, note: str = "") -> str:
        return f"<div class='metric'><span>{esc(title)}</span><strong>{esc(value)}</strong><em>{esc(note)}</em></div>"

    cards = []
    if best is not None:
        cards.append(metric_card("最佳组合", str(best["display_name"]), f"阈值 {best['prob_threshold']:.2f} / 步长 {best['position_step']:.2f}"))
        cards.append(metric_card("最佳累计收益", _format_pct(best["total_return"]), f"年化 {_format_pct(best['annualized_return'])}"))
        cards.append(metric_card("相对买满", _format_pct(best["excess_vs_buy_hold"]), f"买满 {_format_pct(best['buy_hold_total_return'])}"))
    if not baseline_best.empty and best is not None:
        cards.append(metric_card("相对原始规则提升", _format_pct(float(best["total_return"]) - float(baseline_best.iloc[0]["total_return"])), "原始：无阈值、5%步长"))
    cards.append(metric_card("实验数量", str(len(detail)), f"{report['entity_count']} 个模型/集成实体"))

    def rows(df: pd.DataFrame, include_type: bool = True) -> str:
        out = []
        for _, row in df.iterrows():
            cells = [f"<td>{esc(row['display_name'])}</td>"]
            if include_type:
                cells.append(f"<td>{esc(row['entity_type'])}</td>")
            cells.extend(
                [
                    f"<td>{float(row['prob_threshold']):.2f}</td>",
                    f"<td>{float(row['margin_vs_range']):.2f}</td>",
                    f"<td>{float(row['position_step']):.2f}</td>",
                    f"<td>{_format_pct(row['total_return'])}</td>",
                    f"<td>{_format_pct(row['annualized_return'])}</td>",
                    f"<td>{_format_pct(row['max_drawdown'])}</td>",
                    f"<td>{_format_pct(row['active_signal_ratio'])}</td>",
                    f"<td>{_format_pct(row['excess_vs_buy_hold'])}</td>",
                    f"<td>{_format_num(row['sharpe'])}</td>",
                ]
            )
            out.append("<tr>" + "".join(cells) + "</tr>")
        return "\n".join(out)

    sensitivity = (
        detail.groupby(["prob_threshold", "position_step"], dropna=False)
        .agg(avg_total_return=("total_return", "mean"), avg_excess=("excess_vs_buy_hold", "mean"), avg_active=("active_signal_ratio", "mean"), count=("entity_id", "count"))
        .reset_index()
        .sort_values(["avg_total_return", "avg_excess"], ascending=False)
        .head(24)
    )
    sens_rows = []
    for _, row in sensitivity.iterrows():
        sens_rows.append(
            "<tr>"
            f"<td>{float(row['prob_threshold']):.2f}</td>"
            f"<td>{float(row['position_step']):.2f}</td>"
            f"<td>{_format_pct(row['avg_total_return'])}</td>"
            f"<td>{_format_pct(row['avg_excess'])}</td>"
            f"<td>{_format_pct(row['avg_active'])}</td>"
            f"<td>{int(row['count'])}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>不新增数据实验包</title>
<style>
:root {{ --ink:#17202a; --muted:#687385; --line:#d9e0e8; --bg:#f6f8fb; --panel:#fff; --blue:#2868c7; --green:#14785f; --red:#b54747; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:var(--bg); }}
header {{ padding:30px 38px 20px; background:#fff; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:28px; }}
p {{ color:var(--muted); line-height:1.7; margin:0; }}
main {{ padding:24px 38px 44px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-bottom:18px; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-height:92px; }}
.metric span {{ display:block; color:var(--muted); font-size:13px; margin-bottom:8px; }}
.metric strong {{ display:block; font-size:18px; line-height:1.3; }}
.metric em {{ display:block; color:var(--muted); font-style:normal; margin-top:6px; font-size:12px; }}
.section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin-top:18px; }}
h2 {{ margin:0 0 12px; font-size:18px; }}
.note {{ background:#eef5ff; border-left:4px solid var(--blue); padding:12px 14px; line-height:1.7; margin-top:10px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; white-space:nowrap; }}
th {{ background:#f0f3f8; color:#344054; }}
td:first-child {{ min-width:230px; white-space:normal; }}
.footer {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style>
</head>
<body>
<header>
  <h1>不新增数据实验包：阈值、仓位、集成</h1>
  <p>本报告只使用当前已有 CFETS 数据和已训练模型，不引入新数据。目的不是证明模型已经完美，而是看现有信息还能通过决策规则榨出多少。</p>
</header>
<main>
  <div class="metrics">{''.join(cards)}</div>
  <div class="note">解释：概率阈值越高，模型越少出手；仓位步长越高，每次看多/看空调仓越激进；集成是同一期限多个模型概率取平均。</div>
  <section class="section">
    <h2>每个模型/集成的最佳参数</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>模型</th><th>类型</th><th>概率阈值</th><th>相对震荡优势</th><th>仓位步长</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>出手比例</th><th>相对买满</th><th>夏普</th></tr></thead>
      <tbody>{rows(best_entity)}</tbody>
    </table></div>
  </section>
  <section class="section">
    <h2>全局前 20 名参数组合</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>模型</th><th>类型</th><th>概率阈值</th><th>相对震荡优势</th><th>仓位步长</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>出手比例</th><th>相对买满</th><th>夏普</th></tr></thead>
      <tbody>{rows(top)}</tbody>
    </table></div>
  </section>
  <section class="section">
    <h2>平均敏感性</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>概率阈值</th><th>仓位步长</th><th>平均累计收益</th><th>平均相对买满</th><th>平均出手比例</th><th>样本数</th></tr></thead>
      <tbody>{''.join(sens_rows)}</tbody>
    </table></div>
  </section>
  <section class="section">
    <h2>集成模型表现</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>模型</th><th>类型</th><th>概率阈值</th><th>相对震荡优势</th><th>仓位步长</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>出手比例</th><th>相对买满</th><th>夏普</th></tr></thead>
      <tbody>{rows(ensembles) if not ensembles.empty else '<tr><td colspan="11">本次没有可集成的同期限多模型。</td></tr>'}</tbody>
    </table></div>
  </section>
  <div class="footer">明细 CSV：{esc(report['detail_csv'])}；摘要 JSON：{esc(report['summary_json'])}</div>
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def run_strategy_sweep(cfg: StrategySweepConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _read_features(cfg.features_path)
    single_entities = [_load_single_entity(model_dir, features, cfg) for model_dir in cfg.model_dirs]
    entities = single_entities[:]
    if cfg.include_ensembles:
        entities.extend(_make_ensemble_entities(single_entities, out_dir))

    detail_rows = []
    best_details_dir = out_dir / "best_details"
    best_details_dir.mkdir(parents=True, exist_ok=True)
    for entity in entities:
        for threshold in cfg.prob_thresholds:
            for margin in cfg.margins:
                pred = _apply_probability_rule(entity["predictions"], threshold, margin)
                for step in cfg.steps:
                    metrics, strategy_detail = _simulate_strategy(entity, pred, features, cfg, step)
                    detail_rows.append(metrics)

    detail = pd.DataFrame(detail_rows)
    detail_path = out_dir / "strategy_sweep_detail.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    best_by_entity = _best_by_group(detail, ["entity_id"])
    best_by_tenor = _best_by_group(detail, ["target_col", "entity_type"])
    best_entity_path = out_dir / "strategy_sweep_best_by_entity.csv"
    best_tenor_path = out_dir / "strategy_sweep_best_by_tenor.csv"
    best_by_entity.to_csv(best_entity_path, index=False, encoding="utf-8-sig")
    best_by_tenor.to_csv(best_tenor_path, index=False, encoding="utf-8-sig")

    summary_path = out_dir / "no_new_data_experiment_summary.json"
    html_path = out_dir / "no_new_data_experiment_report.html"
    report = {
        "config": {
            **_jsonable_config(cfg),
            "features_path": str(cfg.features_path),
            "model_dirs": [str(path) for path in cfg.model_dirs],
            "out_dir": str(out_dir),
            "duration_map": cfg.duration_map or {},
        },
        "entity_count": len(entities),
        "single_entity_count": len(single_entities),
        "ensemble_entity_count": len(entities) - len(single_entities),
        "detail_csv": str(detail_path),
        "best_by_entity_csv": str(best_entity_path),
        "best_by_tenor_csv": str(best_tenor_path),
        "summary_json": str(summary_path),
        "html_report": str(html_path),
        "detail": detail.to_dict(orient="records"),
        "best_by_entity": best_by_entity.to_dict(orient="records"),
        "best_by_tenor": best_by_tenor.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps({k: v for k, v in report.items() if k not in {"detail"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_strategy_html(report, html_path)
    return {
        "html_report": str(html_path),
        "detail_csv": str(detail_path),
        "best_by_entity_csv": str(best_entity_path),
        "best_by_tenor_csv": str(best_tenor_path),
        "summary_json": str(summary_path),
        "experiment_count": int(len(detail)),
        "entity_count": len(entities),
    }


def _train_config_from_label_cfg(cfg: LabelSweepConfig | RollingValidationConfig, duration: float) -> TrainConfig:
    return TrainConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        patience=cfg.patience,
        device=cfg.device,
        class_weight=cfg.class_weight,
        duration=duration,
        yield_unit=cfg.yield_unit,
    )


def _model_config_from_dataset(dataset, architecture: str, cfg: LabelSweepConfig | RollingValidationConfig) -> ModelConfig:
    return ModelConfig(
        input_size=len(dataset.feature_cols),
        architecture=architecture,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.layers,
        dropout=cfg.dropout,
        n_heads=cfg.heads,
        tcn_kernel_size=cfg.kernel_size,
        max_window=max(512, cfg.window + 8),
    )


def _read_training_metrics(model_dir: Path) -> dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    pred_path = model_dir / "test_predictions.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    report = metrics.get("classification_report", {})
    backtest = metrics.get("backtest", {})
    pred = pd.read_csv(pred_path, encoding="utf-8-sig") if pred_path.exists() else pd.DataFrame()
    return {
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro avg", {}).get("f1-score"),
        "bearish_f1": report.get("看空", report.get("鐪嬬┖", {})).get("f1-score") if isinstance(report.get("看空", report.get("鐪嬬┖", {})), dict) else None,
        "bullish_f1": report.get("看多", report.get("鐪嬪", {})).get("f1-score") if isinstance(report.get("看多", report.get("鐪嬪", {})), dict) else None,
        "range_f1": report.get("震荡", report.get("闇囪崱", {})).get("f1-score") if isinstance(report.get("震荡", report.get("闇囪崱", {})), dict) else None,
        "proxy_total_return": backtest.get("total_return_proxy"),
        "proxy_active_signal_ratio": backtest.get("active_signal_ratio"),
        "proxy_positive_active_ratio": backtest.get("positive_active_ratio"),
        "test_start": str(pred["date"].iloc[0]) if not pred.empty and "date" in pred.columns else "",
        "test_end": str(pred["date"].iloc[-1]) if not pred.empty and "date" in pred.columns else "",
        "test_rows": int(len(pred)),
    }


def _duration_for_target(target_col: str, duration_map: dict[str, float] | None) -> float:
    tenor = _infer_tenor(target_col)
    if tenor and duration_map and tenor in duration_map:
        return float(duration_map[tenor])
    if tenor:
        return max(1.0, float(tenor) * 0.75)
    return 3.0


def _write_simple_training_html(title: str, rows: pd.DataFrame, path: Path, note: str) -> None:
    top = rows.sort_values(["macro_f1", "proxy_total_return"], ascending=False, na_position="last").head(30)

    def esc(value: Any) -> str:
        return html.escape(str(value))

    table_rows = []
    for _, row in top.iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{esc(row.get('target_col', ''))}</td>"
            f"<td>{esc(row.get('fold', ''))}</td>"
            f"<td>{esc(row.get('model', ''))}</td>"
            f"<td>{_format_num(row.get('theta_quantile'))}</td>"
            f"<td>{_format_num(row.get('actual_theta'), 4)}</td>"
            f"<td>{_format_pct(row.get('accuracy'))}</td>"
            f"<td>{_format_pct(row.get('macro_f1'))}</td>"
            f"<td>{_format_pct(row.get('proxy_total_return'))}</td>"
            f"<td>{_format_pct(row.get('proxy_positive_active_ratio'))}</td>"
            f"<td>{esc(row.get('test_start', ''))} 至 {esc(row.get('test_end', ''))}</td>"
            "</tr>"
        )

    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>
body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:#17202a; background:#f6f8fb; }}
header {{ padding:28px 36px 18px; background:#fff; border-bottom:1px solid #d9e0e8; }}
main {{ padding:22px 36px 40px; }}
h1 {{ margin:0 0 8px; font-size:26px; }}
p {{ margin:0; color:#687385; line-height:1.7; }}
.section {{ background:#fff; border:1px solid #d9e0e8; border-radius:8px; padding:18px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid #d9e0e8; padding:9px 10px; text-align:left; white-space:nowrap; }}
th {{ background:#f0f3f8; }}
td:first-child {{ min-width:260px; white-space:normal; }}
</style>
</head>
<body>
<header><h1>{esc(title)}</h1><p>{esc(note)}</p></header>
<main><section class="section"><div class="table-wrap"><table>
<thead><tr><th>目标</th><th>折</th><th>模型</th><th>标签阈值分位</th><th>实际阈值</th><th>准确率</th><th>Macro F1</th><th>方向代理收益</th><th>主动胜率</th><th>测试区间</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table></div></section></main>
</body></html>""",
        encoding="utf-8",
    )


def run_label_threshold_sweep(cfg: LabelSweepConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_col in cfg.target_cols:
        for theta in cfg.theta_quantiles:
            for architecture in cfg.models:
                seed = cfg.seed + len(rows)
                np.random.seed(seed)
                torch.manual_seed(seed)
                dataset = build_windowed_dataset(
                    data_path=cfg.data_path,
                    target_col=target_col,
                    horizon=cfg.horizon,
                    window=cfg.window,
                    theta_quantile=theta,
                    train_end=cfg.train_end,
                    val_end=cfg.val_end,
                    train_ratio=cfg.train_ratio,
                    val_ratio=cfg.val_ratio,
                    exclude_target_feature=cfg.exclude_target_feature,
                    derive_features=cfg.derive_features,
                )
                duration = _duration_for_target(target_col, cfg.duration_map)
                model_dir = out_dir / "models" / _safe_name(target_col) / f"theta_{theta:.2f}" / architecture
                train_model(
                    dataset=dataset,
                    out_dir=model_dir,
                    model_config=_model_config_from_dataset(dataset, architecture, cfg),
                    train_config=_train_config_from_label_cfg(cfg, duration),
                )
                metrics = _read_training_metrics(model_dir)
                rows.append(
                    {
                        "target_col": target_col,
                        "tenor": _infer_tenor(target_col),
                        "model": architecture,
                        "theta_quantile": theta,
                        "actual_theta": dataset.theta,
                        "model_dir": str(model_dir),
                        **metrics,
                    }
                )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "label_threshold_summary.csv"
    html_path = out_dir / "label_threshold_report.html"
    json_path = out_dir / "label_threshold_summary.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps({"config": _jsonable_config(cfg), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_simple_training_html("标签阈值实验", summary, html_path, "同一批数据下改变看多/看空/震荡的划分阈值，观察模型是否更稳。")
    return {
        "html_report": str(html_path),
        "summary_csv": str(summary_path),
        "summary_json": str(json_path),
        "experiment_count": len(rows),
    }


def _rolling_fold_boundaries(raw: pd.DataFrame, cfg: RollingValidationConfig) -> list[dict[str, Any]]:
    n = len(raw)
    folds = []
    train_end_idx = cfg.min_train_rows - 1
    while len(folds) < cfg.max_folds:
        val_end_idx = train_end_idx + cfg.val_rows
        test_end_idx = val_end_idx + cfg.test_rows
        if test_end_idx >= n:
            break
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_end_idx": train_end_idx,
                "val_end_idx": val_end_idx,
                "test_end_idx": test_end_idx,
                "train_end": pd.Timestamp(raw["date"].iloc[train_end_idx]).date().isoformat(),
                "val_end": pd.Timestamp(raw["date"].iloc[val_end_idx]).date().isoformat(),
                "test_end": pd.Timestamp(raw["date"].iloc[test_end_idx]).date().isoformat(),
            }
        )
        train_end_idx += cfg.step_rows
    return folds


def run_rolling_validation(cfg: RollingValidationConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_wide_dataset(cfg.data_path).replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any").reset_index(drop=True)
    if cfg.derive_features:
        raw = add_derived_features(raw)
    folds = _rolling_fold_boundaries(raw, cfg)
    if not folds:
        raise ValueError("当前数据长度不足以生成滚动验证折。请降低 min_train_rows/val_rows/test_rows。")
    fold_data_dir = out_dir / "fold_data"
    fold_data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        fold_data = raw.iloc[: fold["test_end_idx"] + 1].copy()
        fold_path = fold_data_dir / f"fold_{fold['fold']:02d}.csv"
        fold_data.to_csv(fold_path, index=False, encoding="utf-8-sig")
        for target_col in cfg.target_cols:
            for architecture in cfg.models:
                seed = cfg.seed + len(rows)
                np.random.seed(seed)
                torch.manual_seed(seed)
                dataset = build_windowed_dataset(
                    data_path=fold_path,
                    target_col=target_col,
                    horizon=cfg.horizon,
                    window=cfg.window,
                    theta_quantile=cfg.theta_quantile,
                    train_end=fold["train_end"],
                    val_end=fold["val_end"],
                    exclude_target_feature=cfg.exclude_target_feature,
                    derive_features=False,
                )
                duration = _duration_for_target(target_col, cfg.duration_map)
                model_dir = out_dir / "models" / _safe_name(target_col) / f"fold_{fold['fold']:02d}" / architecture
                train_model(
                    dataset=dataset,
                    out_dir=model_dir,
                    model_config=_model_config_from_dataset(dataset, architecture, cfg),
                    train_config=_train_config_from_label_cfg(cfg, duration),
                )
                metrics = _read_training_metrics(model_dir)
                rows.append(
                    {
                        "fold": fold["fold"],
                        "train_end": fold["train_end"],
                        "val_end": fold["val_end"],
                        "fold_test_end": fold["test_end"],
                        "target_col": target_col,
                        "tenor": _infer_tenor(target_col),
                        "model": architecture,
                        "theta_quantile": cfg.theta_quantile,
                        "actual_theta": dataset.theta,
                        "model_dir": str(model_dir),
                        **metrics,
                    }
                )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "rolling_validation_summary.csv"
    html_path = out_dir / "rolling_validation_report.html"
    json_path = out_dir / "rolling_validation_summary.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps({"config": _jsonable_config(cfg), "folds": folds, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_simple_training_html("滚动验证实验", summary, html_path, "用更接近真实投研的方式反复前推训练/验证/测试，观察模型是否只在某一段时间偶然有效。")
    return {
        "html_report": str(html_path),
        "summary_csv": str(summary_path),
        "summary_json": str(json_path),
        "experiment_count": len(rows),
        "fold_count": len(folds),
    }
