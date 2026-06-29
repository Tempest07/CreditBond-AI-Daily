from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.model_arena import _score_predictions
from creditbond_ai.probability_calibration import apply_calibration, calibration_metrics, fit_calibrator
from creditbond_ai.threshold_guard import _score, _search_params, _split_dates, apply_threshold_guard


LABEL_CN = {0: "看空", 1: "看多", 2: "震荡"}
PROB_COLS = ["prob_bearish", "prob_bullish", "prob_range"]
LAYER_PREFIX = {"raw": "raw_eval", "calibrated": "cal_eval", "guard": "cal_guard_eval"}
LAYER_LABEL = {"raw": "原始概率", "calibrated": "概率校准", "guard": "校准+阈值保护"}


@dataclass(frozen=True)
class TenorConfig:
    key: str
    name: str
    out_dir: str
    duration: float


TENORS = [
    TenorConfig("AAA3Y", "3年", "curve_2020_AAA3Y_h5", 2.8),
    TenorConfig("AAA5Y", "5年", "curve_2020_AAA5Y_h5", 4.5),
    TenorConfig("AAA10Y", "10年", "curve_2020_AAA10Y_h5", 7.5),
    TenorConfig("AAAp20Y", "20年", "curve_2020_AAAp20Y_h5", 12.0),
]


@dataclass(frozen=True)
class ScoreTenor:
    name: str
    target_col: str
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validation-selected CreditCurveNet ensemble.")
    parser.add_argument("--models-root", default="models/credit_curve_net_ensemble/v2_antioverfit_ensemble")
    parser.add_argument("--out-dir", default="data/model_arena/curve_2020_credit_curve_net_selected_ensemble_v1")
    parser.add_argument("--min-members", type=int, default=8)
    parser.add_argument("--max-members", type=int, default=28)
    return parser.parse_args()


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "y_true", "future_yield_change", *PROB_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["y_true"].astype(int)
    if "y_pred" not in df.columns:
        df["y_pred"] = df[PROB_COLS].to_numpy(dtype=float).argmax(axis=1)
    df["y_pred"] = df["y_pred"].astype(int)
    df["y_pred_cn"] = [LABEL_CN[int(x)] for x in df["y_pred"]]
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _model_id(path: Path) -> str:
    seed = path.parent.name
    variant = path.parent.parent.name
    return f"{variant}/{seed}"


def _common_dates(frames: list[pd.DataFrame]) -> list[pd.Timestamp]:
    date_sets = [set(frame["date"]) for frame in frames if not frame.empty]
    if not date_sets:
        return []
    return sorted(set.intersection(*date_sets))


def _align(df: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    base = pd.DataFrame({"date": dates})
    out = base.merge(df, on="date", how="left")
    out["y_pred"] = out[PROB_COLS].to_numpy(dtype=float).argmax(axis=1).astype(int)
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in out["y_pred"]]
    return out


def _slice(df: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates}).merge(df, on="date", how="left")


def _score_df(df: pd.DataFrame, tenor: TenorConfig, model: str, family: str = "自研筛选集成") -> dict[str, Any]:
    row, _ = _score_predictions(
        df=df,
        tenor=ScoreTenor(tenor.name, "", tenor.duration),
        model=model,
        family=family,
        source_path="",
    )
    return row


def _selection_score(row: dict[str, Any]) -> float:
    ret = (max(-0.08, min(0.08, float(row["total_return_proxy"]))) + 0.08) / 0.16
    active = float(row["active_ratio"])
    activity = 1.0 - min(1.0, abs(active - 0.55) / 0.55)
    hit = float(row["positive_active_ratio"]) if np.isfinite(float(row["positive_active_ratio"])) else 0.0
    overconf = float(row["overconfident_error_ratio"]) if np.isfinite(float(row["overconfident_error_ratio"])) else 0.0
    return float(0.45 * float(row["macro_f1"]) + 0.20 * hit + 0.20 * ret + 0.10 * activity - 0.05 * overconf)


def _ensemble(frames: list[pd.DataFrame], dates: list[pd.Timestamp], weights: np.ndarray | None = None) -> pd.DataFrame:
    if not frames:
        raise ValueError("没有可集成的模型。")
    aligned = [_align(frame, dates) for frame in frames]
    base = aligned[0][["date", "y_true", "future_yield_change"]].copy()
    stack = np.stack([frame[PROB_COLS].to_numpy(dtype=float) for frame in aligned], axis=0)
    if weights is None:
        probs = np.nanmean(stack, axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        probs = np.tensordot(weights, stack, axes=(0, 0))
    probs = np.clip(np.nan_to_num(probs, nan=1.0 / 3.0), 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1).astype(int)
    out = base.copy()
    out["prob_bearish"] = probs[:, 0]
    out["prob_bullish"] = probs[:, 1]
    out["prob_range"] = probs[:, 2]
    out["y_pred"] = pred
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in pred]
    return out


def _member_table(paths: list[Path], frames: list[pd.DataFrame], tenor: TenorConfig, cal_dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, frame in zip(paths, frames, strict=True):
        cal = _slice(_align(frame, cal_dates), cal_dates)
        row = _score_df(cal, tenor, _model_id(path), "自研子模型")
        row["model_id"] = _model_id(path)
        row["path"] = str(path)
        row["selection_score"] = _selection_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def _candidate_member_counts(total: int, min_members: int, max_members: int) -> list[int]:
    lower = min(max(3, min_members), total)
    upper = min(max_members, total)
    rough = [lower, 10, 12, 16, 20, 24, upper, total]
    return sorted({k for k in rough if lower <= k <= total})


def _evaluate_candidate(
    tenor: TenorConfig,
    model_name: str,
    full_df: pd.DataFrame,
    cal_dates: list[pd.Timestamp],
    eval_dates: list[pd.Timestamp],
) -> tuple[dict[str, Any], pd.DataFrame, Any, Any, dict[str, pd.DataFrame]]:
    cal_raw = _slice(full_df, cal_dates)
    eval_raw = _slice(full_df, eval_dates)
    cal_params, _ = fit_calibrator(cal_raw)
    full_cal = apply_calibration(full_df, cal_params)
    cal_cal = _slice(full_cal, cal_dates)
    eval_cal = _slice(full_cal, eval_dates)
    guard_params, _ = _search_params(
        {"model": model_name, "family": "自研筛选集成", "df": full_cal},
        ScoreTenor(tenor.name, "", tenor.duration),
        cal_dates,
    )
    cal_guard = apply_threshold_guard(cal_cal, guard_params)
    eval_guard = apply_threshold_guard(eval_cal, guard_params)

    cal_score = _score(cal_guard, ScoreTenor(tenor.name, "", tenor.duration), model_name, "自研筛选集成")
    raw_eval_score = _score(eval_raw, ScoreTenor(tenor.name, "", tenor.duration), model_name, "自研筛选集成")
    cal_eval_score = _score(eval_cal, ScoreTenor(tenor.name, "", tenor.duration), model_name, "自研筛选集成")
    guard_eval_score = _score(eval_guard, ScoreTenor(tenor.name, "", tenor.duration), model_name, "自研筛选集成")
    raw_eval_ece = calibration_metrics(eval_raw)["ece"]
    cal_eval_ece = calibration_metrics(eval_cal)["ece"]

    row: dict[str, Any] = {
        "tenor": tenor.name,
        "model": model_name,
        "family": "自研筛选集成",
        "selection_start": pd.to_datetime(cal_dates[0]).strftime("%Y-%m-%d"),
        "selection_end": pd.to_datetime(cal_dates[-1]).strftime("%Y-%m-%d"),
        "evaluation_start": pd.to_datetime(eval_dates[0]).strftime("%Y-%m-%d"),
        "evaluation_end": pd.to_datetime(eval_dates[-1]).strftime("%Y-%m-%d"),
        "selection_objective_score": cal_score["objective_score"],
        "temperature": cal_params.temperature,
        "prior_blend": cal_params.prior_blend,
        "range_bias": cal_params.range_bias,
        "bullish_bias": cal_params.bullish_bias,
        "guard_direction_threshold": guard_params.direction_threshold,
        "guard_direction_margin": guard_params.direction_margin,
        "guard_range_gap": guard_params.range_gap,
        "raw_eval_ece": raw_eval_ece,
        "cal_eval_ece": cal_eval_ece,
    }
    for prefix, metrics in [
        ("raw_eval", raw_eval_score),
        ("cal_eval", cal_eval_score),
        ("cal_guard_eval", guard_eval_score),
    ]:
        for key, value in metrics.items():
            if key in {"tenor", "model", "family", "source_path", "date_start", "date_end"}:
                continue
            row[f"{prefix}_{key}"] = value
    eval_guard = eval_guard.copy()
    eval_guard["tenor"] = tenor.name
    eval_guard["model"] = model_name
    outputs = {
        "raw": eval_raw.copy(),
        "calibrated": eval_cal.copy(),
        "guard": eval_guard.copy(),
    }
    for layer_name, frame in outputs.items():
        frame["tenor"] = tenor.name
        frame["model"] = model_name
        frame["signal_layer"] = layer_name
    return row, eval_guard, cal_params, guard_params, outputs


def _best_layer(row: dict[str, Any] | pd.Series) -> str:
    choices = []
    for layer, prefix in LAYER_PREFIX.items():
        value = float(row[f"{prefix}_objective_score"])
        choices.append((value, layer))
    choices.sort(reverse=True)
    return choices[0][1]


def _attach_selected_metrics(row: dict[str, Any], layer: str) -> dict[str, Any]:
    prefix = LAYER_PREFIX[layer]
    row["chosen_signal_layer"] = layer
    row["chosen_signal_layer_cn"] = LAYER_LABEL[layer]
    for key, value in list(row.items()):
        if key.startswith(prefix + "_"):
            row["selected_eval_" + key.removeprefix(prefix + "_")] = value
    row["selected_eval_ece"] = row["raw_eval_ece"] if layer == "raw" else row["cal_eval_ece"]
    return row


def _run_tenor(
    tenor: TenorConfig,
    models_root: Path,
    min_members: int,
    max_members: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = sorted((models_root / tenor.out_dir).rglob("test_predictions.csv"))
    if len(paths) < min_members:
        raise ValueError(f"{tenor.name} 可用子模型不足: {len(paths)}")
    frames = [_read_prediction(path) for path in paths]
    dates = _common_dates(frames)
    cal_dates, eval_dates = _split_dates(dates, 0.5)
    member_score_dates, topn_tune_dates = _split_dates(cal_dates, 0.5)
    members = _member_table(paths, frames, tenor, member_score_dates)

    sorted_paths = [Path(path) for path in members["path"].tolist()]
    path_to_frame = {str(path): frame for path, frame in zip(paths, frames, strict=True)}
    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = {}
    counts = _candidate_member_counts(len(sorted_paths), min_members, max_members)

    for count in counts:
        selected_paths = sorted_paths[:count]
        selected_frames = [path_to_frame[str(path)] for path in selected_paths]
        full_df = _ensemble(selected_frames, dates)
        row, pred, _, _, _ = _evaluate_candidate(tenor, f"内层筛选Top{count}", full_df, member_score_dates, topn_tune_dates)
        layer = _best_layer(row)
        row["member_count"] = count
        row["total_available_models"] = len(paths)
        row["selection_mode"] = "nested_time_split_equal_weight"
        row["member_score_start"] = pd.to_datetime(member_score_dates[0]).strftime("%Y-%m-%d")
        row["member_score_end"] = pd.to_datetime(member_score_dates[-1]).strftime("%Y-%m-%d")
        row["topn_tune_start"] = pd.to_datetime(topn_tune_dates[0]).strftime("%Y-%m-%d")
        row["topn_tune_end"] = pd.to_datetime(topn_tune_dates[-1]).strftime("%Y-%m-%d")
        row["selected_signal_layer"] = layer
        row["selected_signal_layer_cn"] = LAYER_LABEL[layer]
        row["selection_objective_score"] = row[f"{LAYER_PREFIX[layer]}_objective_score"]
        row["selection_macro_f1"] = row[f"{LAYER_PREFIX[layer]}_macro_f1"]
        row["selection_total_return_proxy"] = row[f"{LAYER_PREFIX[layer]}_total_return_proxy"]
        row["selection_active_ratio"] = row[f"{LAYER_PREFIX[layer]}_active_ratio"]
        candidate_rows.append(row)
        candidate_predictions[count] = (pred, row)

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["selection_objective_score", "cal_guard_eval_macro_f1", "cal_guard_eval_total_return_proxy"],
        ascending=False,
    )
    chosen_count = int(candidates.iloc[0]["member_count"])
    selected_paths = sorted_paths[:chosen_count]
    selected_frames = [path_to_frame[str(path)] for path in selected_paths]
    final_df = _ensemble(selected_frames, dates)
    chosen_layer = str(candidates.iloc[0]["selected_signal_layer"])
    chosen_row, _, _, _, outputs = _evaluate_candidate(tenor, f"筛选集成Top{chosen_count}", final_df, cal_dates, eval_dates)
    chosen_row = _attach_selected_metrics(chosen_row, chosen_layer)
    chosen_pred = outputs[chosen_layer]
    chosen_row["member_count"] = chosen_count
    chosen_row["total_available_models"] = len(paths)
    chosen_row["selection_mode"] = "nested_time_split_equal_weight"
    chosen_row["member_score_start"] = pd.to_datetime(member_score_dates[0]).strftime("%Y-%m-%d")
    chosen_row["member_score_end"] = pd.to_datetime(member_score_dates[-1]).strftime("%Y-%m-%d")
    chosen_row["topn_tune_start"] = pd.to_datetime(topn_tune_dates[0]).strftime("%Y-%m-%d")
    chosen_row["topn_tune_end"] = pd.to_datetime(topn_tune_dates[-1]).strftime("%Y-%m-%d")
    chosen_members = members.head(chosen_count).copy()
    chosen_members["chosen"] = True
    return chosen_row, chosen_pred, chosen_members, candidates


def _diagnosis(summary: pd.DataFrame) -> str:
    avg_raw_f1 = float(summary["raw_eval_macro_f1"].mean())
    avg_selected_f1 = float(summary["selected_eval_macro_f1"].mean())
    avg_raw_ece = float(summary["raw_eval_ece"].mean())
    avg_selected_ece = float(summary["selected_eval_ece"].mean())
    avg_ret = float(summary["selected_eval_total_return_proxy"].mean())
    if avg_selected_f1 > avg_raw_f1 and avg_selected_ece < avg_raw_ece:
        return (
            "筛选集成的主要价值不是把模型做得更激进，而是让它更稳：样本外宏平均 F1 提高，"
            "同时概率校准误差下降。这个方向符合不过拟合约束。"
        )
    if avg_selected_ece < avg_raw_ece:
        return (
            "筛选集成改善了概率可信度，但方向识别还没有同步增强。它更适合作为风控和信号过滤层，"
            "暂时不适合作为唯一交易引擎。"
        )
    if avg_ret > 0:
        return (
            "筛选集成在代理收益上有正贡献，但概率可信度没有全面改善。这个结果需要继续滚动验证，"
            "不能直接升级为生产主模型。"
        )
    return "本轮筛选集成没有通过样本外验收，应该保留实验结果，但不接入日报主信号。"


def _render_report(summary: pd.DataFrame, candidates: pd.DataFrame, meta: dict[str, Any], out_path: Path) -> None:
    avg_raw_f1 = float(summary["raw_eval_macro_f1"].mean())
    avg_selected_f1 = float(summary["selected_eval_macro_f1"].mean())
    avg_raw_ece = float(summary["raw_eval_ece"].mean())
    avg_selected_ece = float(summary["selected_eval_ece"].mean())
    avg_raw_ret = float(summary["raw_eval_total_return_proxy"].mean())
    avg_selected_ret = float(summary["selected_eval_total_return_proxy"].mean())
    conclusion = _diagnosis(summary)

    cards = []
    for _, row in summary.sort_values("tenor").iterrows():
        cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <span>{_esc(row['tenor'])}</span>
                <strong>Top{int(row['member_count'])}/{int(row['total_available_models'])}</strong>
              </div>
              <p class="muted">选择期：{_esc(row['selection_start'])} 至 {_esc(row['selection_end'])}<br>验收期：{_esc(row['evaluation_start'])} 至 {_esc(row['evaluation_end'])}</p>
              <div class="metrics mini">
                <div><span>最终层</span><b>{_esc(row['chosen_signal_layer_cn'])}</b></div>
                <div><span>宏平均 F1</span><b>{_fmt_pct(row['selected_eval_macro_f1'])}</b></div>
                <div><span>代理收益</span><b>{_fmt_pct(row['selected_eval_total_return_proxy'])}</b></div>
                <div><span>活跃信号</span><b>{_fmt_pct(row['selected_eval_active_ratio'])}</b></div>
              </div>
            </article>
            """
        )

    summary_rows = []
    for _, row in summary.sort_values("tenor").iterrows():
        summary_rows.append(
            f"""
            <tr>
              <td>{_esc(row['tenor'])}</td>
              <td>Top{int(row['member_count'])}/{int(row['total_available_models'])}</td>
              <td>{_esc(row['chosen_signal_layer_cn'])}</td>
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['selected_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['raw_eval_ece'])}</td>
              <td>{_fmt_pct(row['selected_eval_ece'])}</td>
              <td>{_fmt_pct(row['raw_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['selected_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['selected_eval_active_ratio'])}</td>
              <td>{_fmt_num(row['guard_direction_threshold'], 2)} / {_fmt_num(row['guard_direction_margin'], 2)} / {_fmt_num(row['guard_range_gap'], 2)}</td>
            </tr>
            """
        )

    candidate_rows = []
    for _, row in candidates.sort_values(["tenor", "selection_objective_score"], ascending=[True, False]).iterrows():
        chosen = "chosen" if int(row["member_count"]) == int(summary.loc[summary["tenor"] == row["tenor"], "member_count"].iloc[0]) else ""
        candidate_rows.append(
            f"""
            <tr class="{chosen}">
              <td>{_esc(row['tenor'])}</td>
              <td>Top{int(row['member_count'])}</td>
              <td>{_esc(row['selected_signal_layer_cn'])}</td>
              <td>{_fmt_num(row['selection_objective_score'], 3)}</td>
              <td>{_fmt_pct(row['selection_macro_f1'])}</td>
              <td>{_fmt_pct(row['selection_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['selection_active_ratio'])}</td>
            </tr>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CreditCurveNet 筛选集成实验报告</title>
  <style>
    :root {{ --ink:#17202c; --muted:#627086; --line:#dbe4ee; --paper:#ffffff; --bg:#eef3f8; --red:#d64b4b; --green:#278d61; --blue:#315f9f; --gold:#b9812a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    .wrap {{ width:min(1180px, calc(100% - 36px)); margin:0 auto; }}
    .hero {{ padding:34px 0 24px; background:linear-gradient(180deg,#fbfcff,#edf3f8); border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--blue); font-weight:800; }}
    h1 {{ margin:8px 0 10px; font-size:38px; line-height:1.12; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:22px; }}
    p {{ line-height:1.75; }}
    .muted,.note {{ color:var(--muted); }}
    .main {{ padding:20px 0 42px; display:grid; gap:16px; }}
    .section,.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 10px 24px rgba(35,48,66,.05); }}
    .section {{ padding:20px; }}
    .takeaway {{ margin:0; font-size:17px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:16px; }}
    .metrics div {{ background:#f6f8fb; border-top:3px solid var(--blue); border-radius:6px; padding:10px 12px; }}
    .metrics span {{ display:block; color:var(--muted); font-size:12px; }}
    .metrics b {{ display:block; margin-top:5px; font-size:18px; }}
    .card-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ padding:16px; }}
    .card-head {{ display:flex; justify-content:space-between; gap:8px; align-items:start; }}
    .card-head span {{ font-size:22px; font-weight:800; }}
    .card-head strong {{ color:var(--red); text-align:right; }}
    .mini {{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .mini div {{ padding:8px; }}
    .mini b {{ font-size:15px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:920px; background:#fff; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid #edf1f5; text-align:left; font-size:13px; white-space:nowrap; }}
    th {{ color:#526173; background:#f7f9fb; font-weight:700; }}
    tr.chosen td {{ background:#fff7e8; }}
    .rules {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .rules div {{ background:#f6f8fb; border-left:4px solid var(--gold); border-radius:6px; padding:10px 12px; }}
    @media (max-width:900px) {{ .card-grid,.rules {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:30px; }} }}
    @media print {{ @page {{ size:A4; margin:8mm; }} body {{ background:#fff; }} .wrap {{ width:100%; }} h1 {{ font-size:26px; }} .section,.card {{ box-shadow:none; break-inside:avoid; }} .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} th,td {{ font-size:10px; padding:6px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 研究辅助 - 自研神经网络筛选集成</div>
      <h1>让模型先通过样本外验收，再谈接入日报</h1>
      <p class="muted">本报告没有用验收期结果挑模型。前半段只负责筛选子模型、校准概率、搜索阈值；后半段只负责验收表现。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{_esc(conclusion)}</p>
      <div class="metrics">
        <div><span>平均宏 F1</span><b>{_fmt_pct(avg_raw_f1)} -> {_fmt_pct(avg_selected_f1)}</b></div>
        <div><span>平均 ECE</span><b>{_fmt_pct(avg_raw_ece)} -> {_fmt_pct(avg_selected_ece)}</b></div>
        <div><span>平均代理收益</span><b>{_fmt_pct(avg_raw_ret)} -> {_fmt_pct(avg_selected_ret)}</b></div>
        <div><span>验收口径</span><b>严格样本外</b></div>
      </div>
    </section>
    <section><div class="card-grid">{''.join(cards)}</div></section>
    <section class="section">
      <h2>分期限验收结果</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>入选子模型</th><th>最终层</th><th>原始宏 F1</th><th>最终宏 F1</th><th>原始 ECE</th><th>最终 ECE</th><th>原始收益</th><th>最终收益</th><th>活跃信号</th><th>阈值 / 边际 / 震荡差</th></tr></thead>
          <tbody>{''.join(summary_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>候选 TopN 选择记录</h2>
      <p class="note">黄色行是最终选中的 TopN。这里的选择分数只来自选择期，不来自验收期。</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>候选</th><th>选择层</th><th>选择期分数</th><th>选择期宏 F1</th><th>选择期收益</th><th>活跃信号</th></tr></thead>
          <tbody>{''.join(candidate_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>防过拟合约束</h2>
      <div class="rules">
        <div><b>时间切分</b><br><span class="muted">前半段选模型，后半段只验收。</span></div>
        <div><b>多模型下限</b><br><span class="muted">每个期限至少 {int(meta['min_members'])} 个子模型，避免押单个随机种子。</span></div>
        <div><b>等权集成</b><br><span class="muted">不按验收期收益加权，减少参数自由度。</span></div>
        <div><b>保留失败</b><br><span class="muted">表现不佳的期限也展示，不隐藏。</span></div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    models_root = Path(args.models_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for tenor in TENORS:
        row, pred, members, candidates = _run_tenor(tenor, models_root, args.min_members, args.max_members)
        summary_rows.append(row)
        prediction_frames.append(pred)
        members["tenor"] = tenor.name
        candidates["tenor"] = tenor.name
        member_frames.append(members)
        candidate_frames.append(candidates)

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    members = pd.concat(member_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    meta = {
        "models_root": str(models_root),
        "min_members": args.min_members,
        "max_members": args.max_members,
        "tenors": [tenor.__dict__ for tenor in TENORS],
    }

    summary_path = out_dir / "selected_ensemble_summary.csv"
    predictions_path = out_dir / "selected_ensemble_predictions.csv"
    members_path = out_dir / "selected_ensemble_members.csv"
    candidates_path = out_dir / "selected_ensemble_candidate_search.csv"
    meta_path = out_dir / "selected_ensemble_meta.json"
    html_path = out_dir / "selected_ensemble_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    members.to_csv(members_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(summary, candidates, meta, html_path)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "predictions": str(predictions_path),
                "members": str(members_path),
                "candidates": str(candidates_path),
                "html": str(html_path),
                "meta": str(meta_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
