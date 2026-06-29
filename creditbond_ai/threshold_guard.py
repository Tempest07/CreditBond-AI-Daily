from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_arena import (
    LABEL_CN,
    TENORS,
    _align_candidate,
    _build_ensemble,
    _common_dates,
    _load_model_candidates,
    _score_predictions,
)


@dataclass(frozen=True)
class GuardParams:
    direction_threshold: float
    direction_margin: float
    range_gap: float


DIRECTION_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
DIRECTION_MARGINS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
RANGE_GAPS = [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15]


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


def _split_dates(dates: list[pd.Timestamp], calibration_ratio: float) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if len(dates) < 40:
        raise ValueError("阈值实验至少需要 40 个共同预测日期。")
    cut = int(len(dates) * calibration_ratio)
    cut = min(max(cut, 20), len(dates) - 20)
    return dates[:cut], dates[cut:]


def apply_threshold_guard(df: pd.DataFrame, params: GuardParams) -> pd.DataFrame:
    out = df.copy()
    p_bear = out["prob_bearish"].astype(float).to_numpy()
    p_bull = out["prob_bullish"].astype(float).to_numpy()
    p_range = out["prob_range"].astype(float).to_numpy()

    direction_prob = np.maximum(p_bear, p_bull)
    direction_label = np.where(p_bear >= p_bull, 0, 1)
    direction_margin = np.abs(p_bear - p_bull)
    direction_vs_range = direction_prob - p_range
    pass_guard = (
        (direction_prob >= params.direction_threshold)
        & (direction_margin >= params.direction_margin)
        & (direction_vs_range >= params.range_gap)
    )

    raw_pred = out["y_pred"].astype(int).to_numpy()
    guarded_pred = np.where(pass_guard, direction_label, 2).astype(int)
    out["raw_y_pred"] = raw_pred
    out["y_pred"] = guarded_pred
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in guarded_pred]
    out["guard_pass"] = pass_guard
    out["direction_prob"] = direction_prob
    out["direction_margin"] = direction_margin
    out["direction_vs_range"] = direction_vs_range
    return out


def _objective(row: dict[str, Any]) -> float:
    active = float(row["active_ratio"])
    if active <= 0.01:
        activity_score = 0.0
    else:
        activity_target = 0.35
        activity_score = 1.0 - min(1.0, abs(active - activity_target) / activity_target)
    ret_component = (max(-0.10, min(0.10, float(row["total_return_proxy"]))) + 0.10) / 0.20
    active_hit = float(row["positive_active_ratio"]) if np.isfinite(float(row["positive_active_ratio"])) else 0.0
    overconf = float(row["overconfident_error_ratio"]) if np.isfinite(float(row["overconfident_error_ratio"])) else 0.0
    return float(
        0.40 * float(row["macro_f1"])
        + 0.22 * active_hit
        + 0.20 * ret_component
        + 0.13 * activity_score
        - 0.05 * overconf
    )


def _slice(candidate: dict[str, Any], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return _align_candidate(candidate, dates)


def _score(df: pd.DataFrame, tenor: Any, model: str, family: str) -> dict[str, Any]:
    row, _ = _score_predictions(df=df, tenor=tenor, model=model, family=family, source_path="")
    row["objective_score"] = _objective(row)
    return row


def _search_params(candidate: dict[str, Any], tenor: Any, calibration_dates: list[pd.Timestamp]) -> tuple[GuardParams, pd.DataFrame]:
    cal = _slice(candidate, calibration_dates)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_params: GuardParams | None = None

    for direction_threshold in DIRECTION_THRESHOLDS:
        for direction_margin in DIRECTION_MARGINS:
            for range_gap in RANGE_GAPS:
                params = GuardParams(direction_threshold, direction_margin, range_gap)
                guarded = apply_threshold_guard(cal, params)
                row = _score(guarded, tenor, candidate["model"], candidate["family"])
                row.update(asdict(params))
                rows.append(row)
                if best_row is None or row["objective_score"] > best_row["objective_score"]:
                    best_row = row
                    best_params = params

    if best_params is None:
        raise ValueError(f"无法为 {tenor.name} {candidate['model']} 搜索阈值。")
    return best_params, pd.DataFrame(rows)


def _candidate_set(models_root: Path, tenor: Any) -> list[dict[str, Any]]:
    raw = _load_model_candidates(models_root, tenor)
    dates = _common_dates(raw)
    if not dates:
        return []
    base = _align_candidate(raw[0], dates)[["date", "y_true", "future_yield_change"]].copy()
    aligned = [{**candidate, "df": _align_candidate(candidate, dates)} for candidate in raw]
    legacy_aligned = [item["df"] for item in aligned if item["family"] == "旧模型"]
    all_aligned = [item["df"] for item in aligned]
    old_ensemble = _build_ensemble("旧模型-概率均值集成", "旧模型集成", legacy_aligned, base)
    all_ensemble = _build_ensemble("新旧-概率均值集成", "混合集成", all_aligned, base)
    for maybe in (old_ensemble, all_ensemble):
        if maybe is not None:
            aligned.append(maybe)
    return aligned


def _diagnosis(summary: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    custom = summary[summary["model"] == "自研-CreditCurveNet"]
    if not custom.empty:
        raw_f1 = float(custom["raw_eval_macro_f1"].mean())
        guard_f1 = float(custom["guard_eval_macro_f1"].mean())
        raw_active = float(custom["raw_eval_active_ratio"].mean())
        guard_active = float(custom["guard_eval_active_ratio"].mean())
        raw_ret = float(custom["raw_eval_total_return_proxy"].mean())
        guard_ret = float(custom["guard_eval_total_return_proxy"].mean())
        notes.append(
            "自研网络平均看，保护层把活跃比例从 "
            f"{_fmt_pct(raw_active)} 降到 {_fmt_pct(guard_active)}，宏 F1 从 {_fmt_pct(raw_f1)} 到 {_fmt_pct(guard_f1)}，"
            f"代理收益从 {_fmt_pct(raw_ret)} 到 {_fmt_pct(guard_ret)}。"
        )
    for tenor, group in summary.groupby("tenor", sort=False):
        custom_one = group[group["model"] == "自研-CreditCurveNet"]
        if custom_one.empty:
            continue
        row = custom_one.iloc[0]
        if float(row["guard_eval_macro_f1"]) > float(row["raw_eval_macro_f1"]):
            notes.append(f"{tenor}：保护层提升了自研网络的样本外宏 F1。")
        elif float(row["guard_eval_active_ratio"]) < float(row["raw_eval_active_ratio"]) * 0.7:
            notes.append(f"{tenor}：保护层主要降低了过度交易，但方向识别还没有真正变强。")
        else:
            notes.append(f"{tenor}：保护层效果有限，说明该期限需要回到模型和标签层继续改。")
    return notes


def _render_report(summary: pd.DataFrame, search: pd.DataFrame, diagnostics: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    custom = summary[summary["model"] == "自研-CreditCurveNet"].copy()
    all_best = summary.sort_values(["guard_eval_objective_score", "guard_eval_macro_f1"], ascending=False).reset_index(drop=True)
    best = all_best.iloc[0]
    date_start = summary["evaluation_start"].min()
    date_end = summary["evaluation_end"].max()
    custom_raw_f1 = float(custom["raw_eval_macro_f1"].mean()) if not custom.empty else 0.0
    custom_guard_f1 = float(custom["guard_eval_macro_f1"].mean()) if not custom.empty else 0.0
    custom_raw_active = float(custom["raw_eval_active_ratio"].mean()) if not custom.empty else 0.0
    custom_guard_active = float(custom["guard_eval_active_ratio"].mean()) if not custom.empty else 0.0

    if custom_guard_f1 > custom_raw_f1:
        conclusion = "震荡保护层对自研网络有正向作用，但仍不能单独证明可以替代旧模型。下一步应把保护层接到滚动训练和多种子集成里。"
    else:
        conclusion = "震荡保护层能压低过度出手，但没有稳定提升自研网络的方向识别。问题不只在阈值，还在模型训练和标签设计。"

    cards = []
    for tenor, group in summary.groupby("tenor", sort=False):
        winner = group.sort_values(["guard_eval_objective_score", "guard_eval_macro_f1"], ascending=False).iloc[0]
        custom_row = group[group["model"] == "自研-CreditCurveNet"]
        custom_text = ""
        if not custom_row.empty:
            row = custom_row.iloc[0]
            custom_text = (
                f"自研网络：宏 F1 {_fmt_pct(row['raw_eval_macro_f1'])} -> {_fmt_pct(row['guard_eval_macro_f1'])}；"
                f"活跃比例 {_fmt_pct(row['raw_eval_active_ratio'])} -> {_fmt_pct(row['guard_eval_active_ratio'])}。"
            )
        cards.append(
            f"""
            <article class="card">
              <div class="card-head"><span>{_html_escape(tenor)}</span><strong>{_html_escape(winner['model'])}</strong></div>
              <p class="muted">验证期：{_html_escape(winner['evaluation_start'])} 至 {_html_escape(winner['evaluation_end'])}</p>
              <div class="metrics mini">
                <div><span>保护后宏 F1</span><b>{_fmt_pct(winner['guard_eval_macro_f1'])}</b></div>
                <div><span>保护后收益</span><b>{_fmt_pct(winner['guard_eval_total_return_proxy'])}</b></div>
                <div><span>活跃比例</span><b>{_fmt_pct(winner['guard_eval_active_ratio'])}</b></div>
                <div><span>阈值</span><b>{_fmt_num(winner['direction_threshold'], 2)}</b></div>
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
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['guard_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['raw_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['guard_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['raw_eval_active_ratio'])}</td>
              <td>{_fmt_pct(row['guard_eval_active_ratio'])}</td>
              <td>{_fmt_num(row['direction_threshold'], 2)} / {_fmt_num(row['direction_margin'], 2)} / {_fmt_num(row['range_gap'], 2)}</td>
            </tr>
            """
        )

    rank_rows = []
    for idx, row in all_best.head(18).iterrows():
        class_name = "custom" if row["model"] == "自研-CreditCurveNet" else ""
        rank_rows.append(
            f"""
            <tr class="{class_name}">
              <td>{idx + 1}</td>
              <td>{_html_escape(row['tenor'])}</td>
              <td>{_html_escape(row['model'])}</td>
              <td>{_html_escape(row['family'])}</td>
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['guard_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['guard_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['guard_eval_active_ratio'])}</td>
              <td>{_fmt_pct(row['guard_eval_positive_active_ratio'])}</td>
              <td>{_fmt_num(row['direction_threshold'], 2)}</td>
              <td>{_fmt_num(row['direction_margin'], 2)}</td>
              <td>{_fmt_num(row['range_gap'], 2)}</td>
            </tr>
            """
        )

    diag_html = "".join(f"<li>{_html_escape(note)}</li>" for note in diagnostics)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 阈值保护层实验</title>
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
    .card-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ padding:16px; }}
    .card-head {{ display:flex; justify-content:space-between; gap:8px; align-items:start; }}
    .card-head span {{ font-size:22px; font-weight:800; }}
    .card-head strong {{ color:var(--red); text-align:right; }}
    .mini {{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .mini div {{ padding:8px; }}
    .mini b {{ font-size:15px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:900px; background:#fff; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid #edf1f5; text-align:left; font-size:13px; white-space:nowrap; }}
    th {{ color:#526173; background:#f7f9fb; font-weight:700; }}
    tr.custom td:nth-child(3), tr.custom td:first-child {{ color:var(--red); font-weight:700; }}
    .diagnosis {{ margin:0; padding-left:22px; line-height:1.8; }}
    code {{ color:#21314a; background:#f1f4f7; padding:2px 5px; border-radius:4px; }}
    @media (max-width:900px) {{
      .card-grid {{ grid-template-columns:1fr; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      h1 {{ font-size:30px; }}
    }}
    @media print {{
      @page {{ size:A4; margin:8mm; }}
      body {{ background:#fff; }}
      .wrap {{ width:100%; }}
      h1 {{ font-size:26px; }}
      .section,.card {{ box-shadow:none; break-inside:avoid; }}
      .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      th,td {{ font-size:10px; padding:6px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 研究辅助 · 震荡保护层</div>
      <h1>阈值搜索：让模型少在震荡市乱出手</h1>
      <p class="muted">方法：每个模型每个期限先用前半段预测样本搜索阈值，再锁定阈值到后半段验证。验证区间为 {date_start} 至 {date_end}。这不是正式交易回测，只是研究辅助实验。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{_html_escape(conclusion)}</p>
      <div class="metrics">
        <div><span>自研原始平均宏 F1</span><b>{_fmt_pct(custom_raw_f1)}</b></div>
        <div><span>自研保护后平均宏 F1</span><b>{_fmt_pct(custom_guard_f1)}</b></div>
        <div><span>自研原始活跃比例</span><b>{_fmt_pct(custom_raw_active)}</b></div>
        <div><span>自研保护后活跃比例</span><b>{_fmt_pct(custom_guard_active)}</b></div>
      </div>
    </section>
    <section><div class="card-grid">{''.join(cards)}</div></section>
    <section class="section">
      <h2>自研网络：保护前后</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>原始宏 F1</th><th>保护后宏 F1</th><th>原始收益</th><th>保护后收益</th><th>原始活跃</th><th>保护后活跃</th><th>阈值 / 边际 / 对震荡优势</th></tr></thead>
          <tbody>{''.join(custom_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>保护后排行榜</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>名次</th><th>期限</th><th>模型</th><th>类型</th><th>原始宏 F1</th><th>保护后宏 F1</th><th>保护后收益</th><th>保护后活跃</th><th>活跃胜率</th><th>阈值</th><th>边际</th><th>对震荡优势</th></tr></thead>
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
      <p class="muted">保护层逻辑：只有当看多/看空方向概率超过阈值、看多与看空概率差距足够大、且方向概率相对震荡概率有足够优势时，才允许输出方向信号，否则降级为震荡。阈值字段依次为 <code>direction_threshold</code>、<code>direction_margin</code>、<code>range_gap</code>。</p>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def run_threshold_guard_experiment(
    models_root: str | Path = "models",
    out_dir: str | Path = "data/model_arena/curve_2020_threshold_guard_v1",
    calibration_ratio: float = 0.5,
) -> dict[str, Path]:
    models_root = Path(models_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    search_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for tenor in TENORS:
        candidates = _candidate_set(models_root, tenor)
        if not candidates:
            continue
        dates = _common_dates(candidates)
        calibration_dates, evaluation_dates = _split_dates(dates, calibration_ratio)
        for candidate in candidates:
            best_params, search_df = _search_params(candidate, tenor, calibration_dates)
            search_df["tenor"] = tenor.name
            search_df["model"] = candidate["model"]
            search_df["family"] = candidate["family"]
            search_rows.append(search_df)

            cal_raw = _slice(candidate, calibration_dates)
            eval_raw = _slice(candidate, evaluation_dates)
            cal_guard = apply_threshold_guard(cal_raw, best_params)
            eval_guard = apply_threshold_guard(eval_raw, best_params)

            raw_cal = _score(cal_raw, tenor, candidate["model"], candidate["family"])
            guard_cal = _score(cal_guard, tenor, candidate["model"], candidate["family"])
            raw_eval = _score(eval_raw, tenor, candidate["model"], candidate["family"])
            guard_eval = _score(eval_guard, tenor, candidate["model"], candidate["family"])

            row: dict[str, Any] = {
                "tenor": tenor.name,
                "model": candidate["model"],
                "family": candidate["family"],
                "calibration_start": pd.to_datetime(calibration_dates[0]).strftime("%Y-%m-%d"),
                "calibration_end": pd.to_datetime(calibration_dates[-1]).strftime("%Y-%m-%d"),
                "evaluation_start": pd.to_datetime(evaluation_dates[0]).strftime("%Y-%m-%d"),
                "evaluation_end": pd.to_datetime(evaluation_dates[-1]).strftime("%Y-%m-%d"),
                **asdict(best_params),
            }
            for prefix, metrics in [
                ("raw_cal", raw_cal),
                ("guard_cal", guard_cal),
                ("raw_eval", raw_eval),
                ("guard_eval", guard_eval),
            ]:
                for key, value in metrics.items():
                    if key in {"tenor", "model", "family", "source_path", "date_start", "date_end"}:
                        continue
                    row[f"{prefix}_{key}"] = value
            row["eval_macro_f1_delta"] = row["guard_eval_macro_f1"] - row["raw_eval_macro_f1"]
            row["eval_return_delta"] = row["guard_eval_total_return_proxy"] - row["raw_eval_total_return_proxy"]
            row["eval_active_delta"] = row["guard_eval_active_ratio"] - row["raw_eval_active_ratio"]
            summary_rows.append(row)

            pred_detail = eval_guard.copy()
            pred_detail["tenor"] = tenor.name
            pred_detail["model"] = candidate["model"]
            pred_detail["family"] = candidate["family"]
            pred_detail["direction_threshold"] = best_params.direction_threshold
            pred_detail["direction_margin"] = best_params.direction_margin
            pred_detail["range_gap"] = best_params.range_gap
            prediction_rows.append(pred_detail)

    if not summary_rows:
        raise ValueError("没有可运行阈值保护层实验的预测文件。")

    summary = pd.DataFrame(summary_rows)
    search = pd.concat(search_rows, ignore_index=True) if search_rows else pd.DataFrame()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    diagnostics = _diagnosis(summary)

    summary_path = out_dir / "threshold_guard_summary.csv"
    search_path = out_dir / "threshold_guard_search.csv"
    predictions_path = out_dir / "threshold_guard_predictions.csv"
    diagnostics_path = out_dir / "threshold_guard_diagnostics.json"
    html_path = out_dir / "threshold_guard_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    search.to_csv(search_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    diagnostics_path.write_text(json.dumps({"diagnostics": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(summary, search, diagnostics, html_path)
    return {
        "summary": summary_path,
        "search": search_path,
        "predictions": predictions_path,
        "diagnostics": diagnostics_path,
        "html": html_path,
    }
