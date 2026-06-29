from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .threshold_guard import (
    GuardParams,
    _candidate_set,
    _score,
    _search_params,
    _slice,
    _split_dates,
    apply_threshold_guard,
)
from .model_arena import LABEL_CN, TENORS, _common_dates


PROB_COLS = ["prob_bearish", "prob_bullish", "prob_range"]


@dataclass(frozen=True)
class CalibrationParams:
    temperature: float
    prior_blend: float
    range_bias: float
    bullish_bias: float
    prior_bearish: float
    prior_bullish: float
    prior_range: float


TEMPERATURES = [0.70, 0.85, 1.00, 1.20, 1.50, 2.00, 2.75, 3.50, 4.50]
PRIOR_BLENDS = [0.00, 0.05, 0.10, 0.18, 0.25, 0.35]
RANGE_BIASES = [-1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
BULLISH_BIASES = [-0.40, 0.00, 0.40]


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _html_escape(value: Any) -> str:
    return html.escape(str(value))


def _safe_probs(df: pd.DataFrame) -> np.ndarray:
    probs = df[PROB_COLS].to_numpy(dtype=float)
    probs = np.nan_to_num(probs, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _class_prior(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=3).astype(float) + 1.0
    return counts / counts.sum()


def calibrate_probs(probs: np.ndarray, params: CalibrationParams) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-6, 1.0)) / params.temperature
    logits[:, 1] += params.bullish_bias
    logits[:, 2] += params.range_bias
    scaled = _softmax(logits)
    prior = np.asarray([params.prior_bearish, params.prior_bullish, params.prior_range], dtype=float)
    out = (1.0 - params.prior_blend) * scaled + params.prior_blend * prior
    out = np.clip(out, 1e-8, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _nll(y: np.ndarray, probs: np.ndarray) -> float:
    chosen = probs[np.arange(len(y)), y.astype(int)]
    return float(-np.mean(np.log(np.clip(chosen, 1e-8, 1.0)))) if len(y) else 0.0


def _brier(y: np.ndarray, probs: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _ece(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y.astype(int)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end == 1.0:
            mask = (conf >= start) & (conf <= end)
        else:
            mask = (conf >= start) & (conf < end)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return total


def _calibration_loss(y: np.ndarray, probs: np.ndarray) -> float:
    return _nll(y, probs) + 0.35 * _brier(y, probs) + 0.35 * _ece(y, probs)


def calibration_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["y_true"].astype(int).to_numpy()
    probs = _safe_probs(df)
    return {
        "nll": _nll(y, probs),
        "brier": _brier(y, probs),
        "ece": _ece(y, probs),
        "avg_confidence": float(probs.max(axis=1).mean()) if len(probs) else 0.0,
    }


def fit_calibrator(df: pd.DataFrame) -> tuple[CalibrationParams, pd.DataFrame]:
    y = df["y_true"].astype(int).to_numpy()
    probs = _safe_probs(df)
    prior = _class_prior(y)
    raw_active = float((probs.argmax(axis=1) != 2).mean()) if len(probs) else 0.0
    min_active = min(0.45, max(0.15, raw_active * 0.35))
    max_active = 0.80
    rows: list[dict[str, Any]] = []
    best_params: CalibrationParams | None = None
    best_loss = float("inf")

    for temperature in TEMPERATURES:
        for prior_blend in PRIOR_BLENDS:
            for range_bias in RANGE_BIASES:
                for bullish_bias in BULLISH_BIASES:
                    params = CalibrationParams(
                        temperature=temperature,
                        prior_blend=prior_blend,
                        range_bias=range_bias,
                        bullish_bias=bullish_bias,
                        prior_bearish=float(prior[0]),
                        prior_bullish=float(prior[1]),
                        prior_range=float(prior[2]),
                    )
                    calibrated = calibrate_probs(probs, params)
                    cal_active = float((calibrated.argmax(axis=1) != 2).mean()) if len(calibrated) else 0.0
                    active_floor_penalty = max(0.0, min_active - cal_active) ** 2
                    active_ceiling_penalty = max(0.0, cal_active - max_active) ** 2
                    metrics = {
                        "temperature": temperature,
                        "prior_blend": prior_blend,
                        "range_bias": range_bias,
                        "bullish_bias": bullish_bias,
                        "raw_active_ratio": raw_active,
                        "calibrated_active_ratio": cal_active,
                        "min_active_ratio": min_active,
                        "nll": _nll(y, calibrated),
                        "brier": _brier(y, calibrated),
                        "ece": _ece(y, calibrated),
                    }
                    metrics["calibration_loss"] = _calibration_loss(y, calibrated)
                    metrics["selection_loss"] = (
                        metrics["calibration_loss"]
                        + 2.5 * active_floor_penalty
                        + 0.8 * active_ceiling_penalty
                    )
                    rows.append(metrics)
                    if metrics["selection_loss"] < best_loss:
                        best_loss = float(metrics["selection_loss"])
                        best_params = params

    if best_params is None:
        raise ValueError("无法拟合概率校准参数。")
    return best_params, pd.DataFrame(rows)


def apply_calibration(df: pd.DataFrame, params: CalibrationParams) -> pd.DataFrame:
    out = df.copy()
    calibrated = calibrate_probs(_safe_probs(out), params)
    out["prob_bearish"] = calibrated[:, 0]
    out["prob_bullish"] = calibrated[:, 1]
    out["prob_range"] = calibrated[:, 2]
    pred = calibrated.argmax(axis=1).astype(int)
    out["y_pred"] = pred
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in pred]
    out["calibration_temperature"] = params.temperature
    out["calibration_prior_blend"] = params.prior_blend
    out["calibration_range_bias"] = params.range_bias
    out["calibration_bullish_bias"] = params.bullish_bias
    return out


def _diagnosis(summary: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    custom = summary[summary["model"] == "自研-CreditCurveNet"].copy()
    if not custom.empty:
        raw_ece = float(custom["raw_eval_ece"].mean())
        cal_ece = float(custom["cal_eval_ece"].mean())
        raw_f1 = float(custom["raw_eval_macro_f1"].mean())
        cal_f1 = float(custom["cal_eval_macro_f1"].mean())
        cal_guard_f1 = float(custom["cal_guard_eval_macro_f1"].mean())
        raw_ret = float(custom["raw_eval_total_return_proxy"].mean())
        cal_guard_ret = float(custom["cal_guard_eval_total_return_proxy"].mean())
        notes.append(
            f"自研网络平均看，ECE 从 {_fmt_pct(raw_ece)} 到 {_fmt_pct(cal_ece)}，"
            f"宏 F1 原始 {_fmt_pct(raw_f1)}，校准后 {_fmt_pct(cal_f1)}，校准加保护后 {_fmt_pct(cal_guard_f1)}。"
        )
        notes.append(
            f"自研网络代理收益原始 {_fmt_pct(raw_ret)}，校准加保护后 {_fmt_pct(cal_guard_ret)}。"
        )
    for tenor, group in summary.groupby("tenor", sort=False):
        custom_one = group[group["model"] == "自研-CreditCurveNet"]
        if custom_one.empty:
            continue
        row = custom_one.iloc[0]
        if float(row["cal_eval_ece"]) < float(row["raw_eval_ece"]):
            notes.append(f"{tenor}：概率校准确实降低了自研网络的校准误差。")
        else:
            notes.append(f"{tenor}：概率校准没有降低自研网络的校准误差，说明分布漂移较明显。")
        if float(row["cal_guard_eval_macro_f1"]) > float(row["raw_eval_macro_f1"]):
            notes.append(f"{tenor}：校准加保护后，方向识别比原始输出更稳。")
        else:
            notes.append(f"{tenor}：校准加保护仍未改善方向识别，需要回到模型训练层。")
    return notes


def _render_report(summary: pd.DataFrame, diagnostics: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    custom = summary[summary["model"] == "自研-CreditCurveNet"].copy()
    ranked = summary.sort_values(["cal_guard_eval_objective_score", "cal_guard_eval_macro_f1"], ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    date_start = summary["evaluation_start"].min()
    date_end = summary["evaluation_end"].max()

    custom_raw_ece = float(custom["raw_eval_ece"].mean()) if not custom.empty else 0.0
    custom_cal_ece = float(custom["cal_eval_ece"].mean()) if not custom.empty else 0.0
    custom_raw_f1 = float(custom["raw_eval_macro_f1"].mean()) if not custom.empty else 0.0
    custom_cal_guard_f1 = float(custom["cal_guard_eval_macro_f1"].mean()) if not custom.empty else 0.0
    custom_raw_ret = float(custom["raw_eval_total_return_proxy"].mean()) if not custom.empty else 0.0
    custom_cal_guard_ret = float(custom["cal_guard_eval_total_return_proxy"].mean()) if not custom.empty else 0.0

    if custom_cal_ece < custom_raw_ece and custom_cal_guard_f1 > custom_raw_f1:
        conclusion = "信号友好概率校准有价值：它让自研网络概率更可信，且配合保护层后方向识别有所改善。下一步适合做多随机种子集成。"
    elif custom_cal_ece < custom_raw_ece:
        conclusion = "信号友好概率校准让自研网络的概率更可信，但还没有稳定改善方向识别。下一步需要模型集成，而不是只调概率。"
    else:
        conclusion = "概率校准没有稳定奏效，说明样本外分布漂移较强。下一步应优先做多种子集成和滚动重训。"

    cards = []
    for tenor, group in summary.groupby("tenor", sort=False):
        winner = group.sort_values(["cal_guard_eval_objective_score", "cal_guard_eval_macro_f1"], ascending=False).iloc[0]
        custom_row = group[group["model"] == "自研-CreditCurveNet"]
        custom_text = ""
        if not custom_row.empty:
            row = custom_row.iloc[0]
            custom_text = (
                f"自研网络：ECE {_fmt_pct(row['raw_eval_ece'])} -> {_fmt_pct(row['cal_eval_ece'])}；"
                f"宏 F1 {_fmt_pct(row['raw_eval_macro_f1'])} -> {_fmt_pct(row['cal_guard_eval_macro_f1'])}。"
            )
        cards.append(
            f"""
            <article class="card">
              <div class="card-head"><span>{_html_escape(tenor)}</span><strong>{_html_escape(winner['model'])}</strong></div>
              <p class="muted">验证期：{_html_escape(winner['evaluation_start'])} 至 {_html_escape(winner['evaluation_end'])}</p>
              <div class="metrics mini">
                <div><span>校准加保护宏 F1</span><b>{_fmt_pct(winner['cal_guard_eval_macro_f1'])}</b></div>
                <div><span>校准加保护收益</span><b>{_fmt_pct(winner['cal_guard_eval_total_return_proxy'])}</b></div>
                <div><span>校准后 ECE</span><b>{_fmt_pct(winner['cal_eval_ece'])}</b></div>
                <div><span>温度</span><b>{_fmt_num(winner['temperature'], 2)}</b></div>
              </div>
              <p class="note">{_html_escape(custom_text)}</p>
            </article>
            """
        )

    custom_rows = []
    for _, row in custom.sort_values("tenor").iterrows():
        custom_rows.append(
            f"""
            <tr>
              <td>{_html_escape(row['tenor'])}</td>
              <td>{_fmt_pct(row['raw_eval_ece'])}</td>
              <td>{_fmt_pct(row['cal_eval_ece'])}</td>
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['cal_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['raw_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_total_return_proxy'])}</td>
              <td>{_fmt_num(row['temperature'], 2)} / {_fmt_num(row['prior_blend'], 2)} / {_fmt_num(row['range_bias'], 2)}</td>
            </tr>
            """
        )

    rank_rows = []
    for idx, row in ranked.head(18).iterrows():
        class_name = "custom" if row["model"] == "自研-CreditCurveNet" else ""
        rank_rows.append(
            f"""
            <tr class="{class_name}">
              <td>{idx + 1}</td>
              <td>{_html_escape(row['tenor'])}</td>
              <td>{_html_escape(row['model'])}</td>
              <td>{_html_escape(row['family'])}</td>
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['cal_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['raw_eval_ece'])}</td>
              <td>{_fmt_pct(row['cal_eval_ece'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_active_ratio'])}</td>
            </tr>
            """
        )

    diag_html = "".join(f"<li>{_html_escape(note)}</li>" for note in diagnostics)
    beginner_cards = f"""
      <article class="explain-card">
        <span>1</span>
        <h3>先看 ECE：概率有没有骗人</h3>
        <p>ECE 可以理解成“模型自信程度和真实命中率之间的偏差”。如果模型经常说 80% 把握，但实际只对 50%，ECE 就会很高。越低越好，0% 是理想状态。</p>
        <b>本次自研网络：{_fmt_pct(custom_raw_ece)} -> {_fmt_pct(custom_cal_ece)}</b>
      </article>
      <article class="explain-card">
        <span>2</span>
        <h3>再看宏 F1：方向识别是否均衡</h3>
        <p>宏 F1 会同时考察看空、看多、震荡三类，不会因为模型只会猜“震荡”就给它太高评价。越高越好，更适合看这个三分类问题。</p>
        <b>本次自研网络：{_fmt_pct(custom_raw_f1)} -> {_fmt_pct(custom_cal_guard_f1)}</b>
      </article>
      <article class="explain-card">
        <span>3</span>
        <h3>最后看代理收益：只是研究模拟</h3>
        <p>代理收益用“收益率变化 × 久期方向”粗略估算，不含真实组合、流动性、交易成本和持仓约束。它能帮助比较模型，但不能当作真实收益承诺。</p>
        <b>本次自研网络：{_fmt_pct(custom_raw_ret)} -> {_fmt_pct(custom_cal_guard_ret)}</b>
      </article>
    """
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 概率校准实验</title>
  <style>
    :root {{ --ink:#17202c; --muted:#5d6b7c; --line:#dfe6ee; --soft:#f5f7fa; --red:#d94848; --blue:#425f9c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:#eef2f6; }}
    .wrap {{ width:min(1180px, calc(100% - 36px)); margin:0 auto; }}
    .hero {{ padding:34px 0 24px; background:linear-gradient(180deg,#fbfcfd,#eef2f6); border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--blue); font-weight:700; }}
    h1 {{ margin:8px 0 10px; font-size:38px; line-height:1.12; }}
    p {{ line-height:1.7; }}
    .muted,.note {{ color:var(--muted); }}
    .main {{ padding:20px 0 42px; display:grid; gap:16px; }}
    .section,.card {{ background:#fff; border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 22px rgba(35,48,66,.05); }}
    .section {{ padding:20px; }}
    h2 {{ margin:0 0 14px; font-size:22px; }}
    .takeaway {{ font-size:17px; line-height:1.75; margin:0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:16px; }}
    .metrics div {{ background:var(--soft); border-top:3px solid var(--blue); border-radius:6px; padding:10px 12px; }}
    .metrics span {{ display:block; color:var(--muted); font-size:12px; }}
    .metrics b {{ display:block; margin-top:5px; font-size:18px; }}
    .guide-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .explain-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 8px 22px rgba(35,48,66,.05); }}
    .explain-card span {{ display:inline-grid; place-items:center; width:28px; height:28px; border-radius:50%; background:#e9eef7; color:var(--blue); font-weight:800; }}
    .explain-card h3 {{ margin:10px 0 6px; font-size:18px; }}
    .explain-card p {{ margin:0; color:var(--muted); font-size:14px; line-height:1.65; }}
    .explain-card b {{ display:block; margin-top:10px; color:var(--red); font-size:16px; }}
    .read-path {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .read-path div {{ border-left:3px solid var(--blue); background:var(--soft); border-radius:6px; padding:10px 12px; }}
    .read-path span {{ display:block; color:var(--muted); font-size:12px; }}
    .read-path b {{ display:block; margin-top:5px; }}
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
    tr.custom td:nth-child(3), tr.custom td:first-child {{ color:var(--red); font-weight:700; }}
    .diagnosis {{ margin:0; padding-left:22px; line-height:1.8; }}
    code {{ color:#21314a; background:#f1f4f7; padding:2px 5px; border-radius:4px; }}
    @media (max-width:900px) {{
      .card-grid,.guide-grid,.read-path {{ grid-template-columns:1fr; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      h1 {{ font-size:30px; }}
    }}
    @media print {{
      @page {{ size:A4; margin:8mm; }}
      body {{ background:#fff; }}
      .wrap {{ width:100%; }}
      h1 {{ font-size:26px; }}
      .section,.card {{ box-shadow:none; break-inside:avoid; }}
      .guide-grid {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
      .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      th,td {{ font-size:10px; padding:6px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 研究辅助 · 概率校准</div>
      <h1>先让概率可信，再决定要不要出手</h1>
      <p class="muted">方法：前半段预测样本拟合信号友好校准参数和保护层阈值，后半段验证。验证区间为 {date_start} 至 {date_end}。ECE 越低，说明模型说出的概率越接近真实胜率；信号友好约束用于避免概率全部坍缩成震荡。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{_html_escape(conclusion)}</p>
      <div class="metrics">
        <div><span>概率偏差 ECE，越低越好</span><b>{_fmt_pct(custom_raw_ece)} -> {_fmt_pct(custom_cal_ece)}</b></div>
        <div><span>方向识别 宏 F1，越高越好</span><b>{_fmt_pct(custom_raw_f1)} -> {_fmt_pct(custom_cal_guard_f1)}</b></div>
        <div><span>研究代理收益</span><b>{_fmt_pct(custom_raw_ret)} -> {_fmt_pct(custom_cal_guard_ret)}</b></div>
        <div><span>验证区间</span><b>{date_start} 至 {date_end}</b></div>
      </div>
    </section>
    <section class="guide-grid">
      {beginner_cards}
    </section>
    <section class="section">
      <h2>新手阅读顺序</h2>
      <div class="read-path">
        <div><span>第一步</span><b>看 ECE 是否下降</b></div>
        <div><span>第二步</span><b>看宏 F1 是否上升</b></div>
        <div><span>第三步</span><b>看收益有没有被牺牲太多</b></div>
        <div><span>第四步</span><b>看分期限是否稳定</b></div>
      </div>
    </section>
    <section><div class="card-grid">{''.join(cards)}</div></section>
    <section class="section">
      <h2>自研网络：校准前后</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>原始 ECE</th><th>校准 ECE</th><th>原始宏 F1</th><th>校准宏 F1</th><th>校准加保护宏 F1</th><th>原始收益</th><th>校准加保护收益</th><th>温度 / 收缩 / 震荡偏置</th></tr></thead>
          <tbody>{''.join(custom_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>校准加保护排行榜</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>名次</th><th>期限</th><th>模型</th><th>类型</th><th>原始宏 F1</th><th>校准宏 F1</th><th>校准加保护宏 F1</th><th>原始 ECE</th><th>校准 ECE</th><th>校准加保护收益</th><th>活跃比例</th></tr></thead>
          <tbody>{''.join(rank_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>诊断</h2>
      <ul class="diagnosis">{diag_html}</ul>
    </section>
    <section class="section">
      <h2>口径</h2>
      <p class="muted">校准搜索包括 <code>temperature</code>、<code>prior_blend</code>、<code>range_bias</code> 和 <code>bullish_bias</code>。校准只用前半段样本拟合，后半段只做验证。选择校准参数时加入最低方向活跃约束，避免纯校准为了降低 ECE 而全部压成震荡。校准加保护是在校准后的概率上重新搜索并锁定保护层阈值。</p>
      <p class="muted"><b>术语翻译：</b> ECE = 概率可信度误差，越低越好；宏 F1 = 看空/看多/震荡三类的均衡识别分数，越高越好；温度 = 给模型概率“降温”的幅度，越大通常越不自信；收缩 = 把模型概率向历史类别分布拉回一点；震荡偏置 = 让模型更谨慎、更愿意承认震荡。</p>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def run_probability_calibration_experiment(
    models_root: str | Path = "models",
    out_dir: str | Path = "data/model_arena/curve_2020_probability_calibration_v1",
    calibration_ratio: float = 0.5,
) -> dict[str, Path]:
    models_root = Path(models_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    calibrator_rows: list[pd.DataFrame] = []
    guard_search_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for tenor in TENORS:
        candidates = _candidate_set(models_root, tenor)
        if not candidates:
            continue
        dates = _common_dates(candidates)
        calibration_dates, evaluation_dates = _split_dates(dates, calibration_ratio)
        for candidate in candidates:
            raw_cal = _slice(candidate, calibration_dates)
            raw_eval = _slice(candidate, evaluation_dates)
            params, calibrator_search = fit_calibrator(raw_cal)
            calibrator_search["tenor"] = tenor.name
            calibrator_search["model"] = candidate["model"]
            calibrator_search["family"] = candidate["family"]
            calibrator_rows.append(calibrator_search)

            cal_cal = apply_calibration(raw_cal, params)
            cal_eval = apply_calibration(raw_eval, params)
            calibrated_candidate = {**candidate, "df": cal_cal}
            guard_params, guard_search = _search_params(calibrated_candidate, tenor, calibration_dates)
            guard_search["tenor"] = tenor.name
            guard_search["model"] = candidate["model"]
            guard_search["family"] = candidate["family"]
            guard_search_rows.append(guard_search)
            cal_guard_eval = apply_threshold_guard(cal_eval, guard_params)

            raw_eval_score = _score(raw_eval, tenor, candidate["model"], candidate["family"])
            cal_eval_score = _score(cal_eval, tenor, candidate["model"], candidate["family"])
            cal_guard_eval_score = _score(cal_guard_eval, tenor, candidate["model"], candidate["family"])
            raw_cal_metrics = calibration_metrics(raw_cal)
            cal_cal_metrics = calibration_metrics(cal_cal)
            raw_eval_metrics = calibration_metrics(raw_eval)
            cal_eval_metrics = calibration_metrics(cal_eval)

            row: dict[str, Any] = {
                "tenor": tenor.name,
                "model": candidate["model"],
                "family": candidate["family"],
                "calibration_start": pd.to_datetime(calibration_dates[0]).strftime("%Y-%m-%d"),
                "calibration_end": pd.to_datetime(calibration_dates[-1]).strftime("%Y-%m-%d"),
                "evaluation_start": pd.to_datetime(evaluation_dates[0]).strftime("%Y-%m-%d"),
                "evaluation_end": pd.to_datetime(evaluation_dates[-1]).strftime("%Y-%m-%d"),
                **asdict(params),
                "guard_direction_threshold": guard_params.direction_threshold,
                "guard_direction_margin": guard_params.direction_margin,
                "guard_range_gap": guard_params.range_gap,
            }
            for prefix, metrics in [
                ("raw_eval", raw_eval_score),
                ("cal_eval", cal_eval_score),
                ("cal_guard_eval", cal_guard_eval_score),
            ]:
                for key, value in metrics.items():
                    if key in {"tenor", "model", "family", "source_path", "date_start", "date_end"}:
                        continue
                    row[f"{prefix}_{key}"] = value
            for prefix, metrics in [
                ("raw_cal", raw_cal_metrics),
                ("cal_cal", cal_cal_metrics),
                ("raw_eval", raw_eval_metrics),
                ("cal_eval", cal_eval_metrics),
            ]:
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            row["eval_ece_delta"] = row["cal_eval_ece"] - row["raw_eval_ece"]
            row["eval_macro_f1_delta"] = row["cal_guard_eval_macro_f1"] - row["raw_eval_macro_f1"]
            row["eval_return_delta"] = row["cal_guard_eval_total_return_proxy"] - row["raw_eval_total_return_proxy"]
            summary_rows.append(row)

            pred_detail = cal_guard_eval.copy()
            pred_detail["tenor"] = tenor.name
            pred_detail["model"] = candidate["model"]
            pred_detail["family"] = candidate["family"]
            pred_detail["temperature"] = params.temperature
            pred_detail["prior_blend"] = params.prior_blend
            pred_detail["range_bias"] = params.range_bias
            pred_detail["bullish_bias"] = params.bullish_bias
            pred_detail["guard_direction_threshold"] = guard_params.direction_threshold
            pred_detail["guard_direction_margin"] = guard_params.direction_margin
            pred_detail["guard_range_gap"] = guard_params.range_gap
            prediction_rows.append(pred_detail)

    if not summary_rows:
        raise ValueError("没有可运行概率校准实验的预测文件。")

    summary = pd.DataFrame(summary_rows)
    calibrator_search = pd.concat(calibrator_rows, ignore_index=True) if calibrator_rows else pd.DataFrame()
    guard_search = pd.concat(guard_search_rows, ignore_index=True) if guard_search_rows else pd.DataFrame()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    diagnostics = _diagnosis(summary)

    summary_path = out_dir / "probability_calibration_summary.csv"
    calibrator_search_path = out_dir / "probability_calibration_search.csv"
    guard_search_path = out_dir / "probability_calibration_guard_search.csv"
    predictions_path = out_dir / "probability_calibration_predictions.csv"
    diagnostics_path = out_dir / "probability_calibration_diagnostics.json"
    html_path = out_dir / "probability_calibration_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    calibrator_search.to_csv(calibrator_search_path, index=False, encoding="utf-8-sig")
    guard_search.to_csv(guard_search_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    diagnostics_path.write_text(json.dumps({"diagnostics": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(summary, diagnostics, html_path)
    return {
        "summary": summary_path,
        "calibrator_search": calibrator_search_path,
        "guard_search": guard_search_path,
        "predictions": predictions_path,
        "diagnostics": diagnostics_path,
        "html": html_path,
    }
