from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data import load_wide_dataset


LABEL_CN = {0: "看空", 1: "看多", 2: "震荡"}
LABEL_TO_SIGNAL = {0: -1, 1: 1, 2: 0}


@dataclass(frozen=True)
class ArenaTenor:
    name: str
    target_col: str
    duration: float
    legacy_dir: str
    custom_dir: str


TENORS = [
    ArenaTenor(
        name="3年",
        target_col="中债中短期票据到期收益率(AAA):3年",
        duration=2.8,
        legacy_dir="curve_2020_AAA3Y_h5",
        custom_dir="curve_2020_AAA3Y_h5",
    ),
    ArenaTenor(
        name="5年",
        target_col="中债中短期票据到期收益率(AAA):5年",
        duration=4.5,
        legacy_dir="curve_2020_AAA5Y_h5",
        custom_dir="curve_2020_AAA5Y_h5",
    ),
    ArenaTenor(
        name="10年",
        target_col="中债中短期票据到期收益率(AAA):10年",
        duration=7.5,
        legacy_dir="curve_2020_AAA10Y_h5",
        custom_dir="curve_2020_AAA10Y_h5",
    ),
    ArenaTenor(
        name="20年",
        target_col="中债中短期票据到期收益率(AAA+):20年",
        duration=12.0,
        legacy_dir="curve_2020_AAAp20Y_h5",
        custom_dir="curve_2020_AAAp20Y_h5",
    ),
]


def _read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "y_true", "y_pred", "future_yield_change"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    for col in ("prob_bearish", "prob_bullish", "prob_range"):
        if col not in df.columns:
            df[col] = np.nan
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _load_model_candidates(models_root: Path, tenor: ArenaTenor) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for arch in ("gru", "tcn", "transformer"):
        path = models_root / tenor.legacy_dir / "01_full_features" / arch / "test_predictions.csv"
        if path.exists():
            candidates.append(
                {
                    "model": f"旧模型-{arch.upper() if arch != 'tcn' else 'TCN'}",
                    "family": "旧模型",
                    "path": path,
                    "df": _read_prediction(path),
                }
            )
    custom_path = models_root / "credit_curve_net" / tenor.custom_dir / "v1_custom_credit_curve_net" / "test_predictions.csv"
    if custom_path.exists():
        candidates.append(
            {
                "model": "自研-CreditCurveNet",
                "family": "自研神经网络",
                "path": custom_path,
                "df": _read_prediction(custom_path),
            }
        )
    return candidates


def _common_dates(candidates: list[dict[str, Any]]) -> list[pd.Timestamp]:
    date_sets = [set(item["df"]["date"]) for item in candidates if not item["df"].empty]
    if not date_sets:
        return []
    common = set.intersection(*date_sets)
    return sorted(common)


def _align_candidate(candidate: dict[str, Any], dates: list[pd.Timestamp]) -> pd.DataFrame:
    wanted = pd.DataFrame({"date": dates})
    aligned = wanted.merge(candidate["df"], on="date", how="left")
    return aligned


def _build_ensemble(
    name: str,
    family: str,
    dfs: list[pd.DataFrame],
    base: pd.DataFrame,
) -> dict[str, Any] | None:
    if not dfs:
        return None
    prob_cols = ["prob_bearish", "prob_bullish", "prob_range"]
    usable = []
    for df in dfs:
        if all(col in df.columns for col in prob_cols) and not df[prob_cols].isna().all().all():
            usable.append(df[["date", *prob_cols]].copy())
    if not usable:
        return None
    merged = base[["date", "y_true", "future_yield_change"]].copy()
    prob_stack = []
    for df in usable:
        one = merged[["date"]].merge(df, on="date", how="left")
        prob_stack.append(one[prob_cols].to_numpy(dtype=float))
    probs = np.nanmean(np.stack(prob_stack, axis=0), axis=0)
    pred = np.nanargmax(probs, axis=1).astype(int)
    out = merged.copy()
    out["y_pred"] = pred
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in pred]
    out["prob_bearish"] = probs[:, 0]
    out["prob_bullish"] = probs[:, 1]
    out["prob_range"] = probs[:, 2]
    return {"model": name, "family": family, "path": "", "df": out}


def _load_features(features_path: Path) -> pd.DataFrame:
    features = load_wide_dataset(features_path)
    features["date"] = pd.to_datetime(features["date"])
    return features.sort_values("date").replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")


def _baseline_predictions(
    features: pd.DataFrame,
    tenor: ArenaTenor,
    base: pd.DataFrame,
    window: int = 5,
) -> list[dict[str, Any]]:
    if tenor.target_col not in features.columns:
        return []
    series = features[["date", tenor.target_col]].copy().sort_values("date")
    series["past_change"] = series[tenor.target_col].diff(window)
    train_cut = pd.to_datetime(base["date"]).min()
    threshold_source = series.loc[series["date"] < train_cut, "past_change"].abs().dropna()
    threshold = float(threshold_source.quantile(0.6)) if not threshold_source.empty else 0.0
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(series["past_change"].abs().dropna().quantile(0.6))
    if not np.isfinite(threshold):
        threshold = 0.0

    aligned = base[["date", "y_true", "future_yield_change"]].merge(
        series[["date", "past_change"]],
        on="date",
        how="left",
    )
    past = aligned["past_change"].fillna(0.0).to_numpy(dtype=float)
    momentum = np.select([past > threshold, past < -threshold], [0, 1], default=2).astype(int)
    reversal = np.select([past > threshold, past < -threshold], [1, 0], default=2).astype(int)
    always_range = np.full(len(aligned), 2, dtype=int)
    always_bullish = np.full(len(aligned), 1, dtype=int)

    def make(name: str, family: str, pred: np.ndarray) -> dict[str, Any]:
        df = aligned[["date", "y_true", "future_yield_change"]].copy()
        df["y_pred"] = pred
        df["y_pred_cn"] = [LABEL_CN[int(x)] for x in pred]
        df["prob_bearish"] = np.where(pred == 0, 1.0, 0.0)
        df["prob_bullish"] = np.where(pred == 1, 1.0, 0.0)
        df["prob_range"] = np.where(pred == 2, 1.0, 0.0)
        return {"model": name, "family": family, "path": "", "df": df}

    return [
        make("基准-永远震荡", "基准", always_range),
        make("基准-永远看多", "基准", always_bullish),
        make("基准-收益率动量", "基准", momentum),
        make("基准-收益率反转", "基准", reversal),
    ]


def _annualized_return(total_return: float, periods: int, annual_days: int = 252) -> float:
    if periods <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (annual_days / periods) - 1.0)


def _max_drawdown(cum_returns: pd.Series) -> float:
    wealth = 1.0 + cum_returns.fillna(0.0)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def _balanced_score(row: dict[str, Any]) -> float:
    ret_component = (max(-0.10, min(0.10, float(row["total_return_proxy"]))) + 0.10) / 0.20
    active_hit = float(row["positive_active_ratio"]) if np.isfinite(float(row["positive_active_ratio"])) else 0.0
    return float(0.50 * row["macro_f1"] + 0.25 * active_hit + 0.25 * ret_component)


def _score_predictions(
    df: pd.DataFrame,
    tenor: ArenaTenor,
    model: str,
    family: str,
    source_path: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work = df.copy()
    y_true = work["y_true"].astype(int).to_numpy()
    y_pred = work["y_pred"].astype(int).to_numpy()
    signals = pd.Series(y_pred).map(LABEL_TO_SIGNAL).fillna(0).to_numpy(dtype=float)
    changes = work["future_yield_change"].to_numpy(dtype=float)
    pnl = -tenor.duration * signals * (changes / 100.0)
    cum = np.cumsum(pnl)
    active = signals != 0
    probs = work[["prob_bearish", "prob_bullish", "prob_range"]].to_numpy(dtype=float)
    has_prob = np.isfinite(probs).any()
    max_prob = np.nanmax(probs, axis=1) if has_prob else np.full(len(work), np.nan)
    wrong = y_true != y_pred

    row = {
        "tenor": tenor.name,
        "model": model,
        "family": family,
        "source_path": str(source_path),
        "n": int(len(work)),
        "date_start": pd.to_datetime(work["date"]).min().strftime("%Y-%m-%d") if len(work) else "",
        "date_end": pd.to_datetime(work["date"]).max().strftime("%Y-%m-%d") if len(work) else "",
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(work) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)) if len(work) else 0.0,
        "bearish_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)) if len(work) else 0.0,
        "bullish_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)) if len(work) else 0.0,
        "range_f1": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)) if len(work) else 0.0,
        "active_ratio": float(active.mean()) if len(work) else 0.0,
        "bearish_signal_ratio": float((y_pred == 0).mean()) if len(work) else 0.0,
        "bullish_signal_ratio": float((y_pred == 1).mean()) if len(work) else 0.0,
        "range_signal_ratio": float((y_pred == 2).mean()) if len(work) else 0.0,
        "total_return_proxy": float(cum[-1]) if len(cum) else 0.0,
        "annualized_return_proxy": _annualized_return(float(cum[-1]) if len(cum) else 0.0, len(work)),
        "max_drawdown_proxy": _max_drawdown(pd.Series(cum)),
        "mean_active_return_proxy": float(np.mean(pnl[active])) if active.any() else 0.0,
        "positive_active_ratio": float((pnl[active] > 0).mean()) if active.any() else 0.0,
        "avg_confidence": float(np.nanmean(max_prob)) if has_prob else np.nan,
        "overconfident_error_ratio": float(((max_prob >= 0.60) & wrong).mean()) if has_prob else np.nan,
    }
    row["balanced_score"] = _balanced_score(row)

    detail = work[["date", "y_true", "y_pred", "future_yield_change", "prob_bearish", "prob_bullish", "prob_range"]].copy()
    detail["tenor"] = tenor.name
    detail["model"] = model
    detail["family"] = family
    detail["signal"] = signals
    detail["pnl_proxy"] = pnl
    detail["cum_pnl_proxy"] = cum
    detail["max_prob"] = max_prob
    return row, detail


def _diagnose_custom(summary: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for tenor, group in summary.groupby("tenor", sort=False):
        custom = group[group["model"] == "自研-CreditCurveNet"]
        if custom.empty:
            notes.append(f"{tenor}：没有找到自研模型产物。")
            continue
        c = custom.iloc[0]
        legacy = group[group["family"].isin(["旧模型", "旧模型集成"])].sort_values(
            ["balanced_score", "macro_f1"],
            ascending=False,
        )
        best = legacy.iloc[0] if not legacy.empty else None
        pieces = []
        if best is not None:
            if float(c["macro_f1"]) + 0.05 < float(best["macro_f1"]):
                pieces.append(f"样本外 F1 明显落后于 {best['model']}")
            if float(c["active_ratio"]) > float(best["active_ratio"]) + 0.25:
                pieces.append("方向信号过多，容易把震荡行情误判成趋势")
        if float(c["range_f1"]) < 0.20:
            pieces.append("震荡识别很弱，这是当前最大问题")
        if np.isfinite(float(c["avg_confidence"])) and float(c["avg_confidence"]) > 0.70 and float(c["macro_f1"]) < 0.25:
            pieces.append("概率偏自信，需要做校准或阈值层")
        if float(c["total_return_proxy"]) < 0:
            pieces.append("方向代理收益为负，不能接入正式日报")
        if not pieces:
            pieces.append("表现接近可观察区间，但仍需滚动训练确认")
        notes.append(f"{tenor}：" + "；".join(pieces) + "。")
    return notes


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


def _bar(value: float, color: str = "#d84a4a") -> str:
    width = max(0.0, min(100.0, float(value) * 100.0 if np.isfinite(float(value)) else 0.0))
    return f'<span class="bar"><i style="width:{width:.1f}%;background:{color}"></i></span>'


def _html_escape(value: Any) -> str:
    return html.escape(str(value))


def _render_report(summary: pd.DataFrame, details: pd.DataFrame, diagnostics: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        raise ValueError("summary is empty.")
    ranked = summary.sort_values(["balanced_score", "macro_f1", "total_return_proxy"], ascending=False).reset_index(drop=True)
    date_start = summary["date_start"].min()
    date_end = summary["date_end"].max()
    n_models = int(summary["model"].nunique())
    best = ranked.iloc[0]
    custom_rows = summary[summary["model"] == "自研-CreditCurveNet"].copy()
    custom_avg_f1 = float(custom_rows["macro_f1"].mean()) if not custom_rows.empty else 0.0
    old_rows = summary[summary["family"].isin(["旧模型", "旧模型集成"])].copy()
    old_avg_f1 = float(old_rows["macro_f1"].mean()) if not old_rows.empty else 0.0

    cards = []
    for tenor, group in summary.groupby("tenor", sort=False):
        winner = group.sort_values(["balanced_score", "macro_f1", "total_return_proxy"], ascending=False).iloc[0]
        custom = group[group["model"] == "自研-CreditCurveNet"].iloc[0] if not group[group["model"] == "自研-CreditCurveNet"].empty else None
        cards.append(
            f"""
            <article class="card tenor-card">
              <div class="card-head"><span>{_html_escape(tenor)}</span><strong>{_html_escape(winner['model'])}</strong></div>
              <p class="muted">统一区间：{_html_escape(winner['date_start'])} 至 {_html_escape(winner['date_end'])}</p>
              <div class="metrics mini">
                <div><span>综合分</span><b>{_fmt_num(winner['balanced_score'])}</b></div>
                <div><span>宏 F1</span><b>{_fmt_pct(winner['macro_f1'])}</b></div>
                <div><span>代理收益</span><b>{_fmt_pct(winner['total_return_proxy'])}</b></div>
                <div><span>活跃胜率</span><b>{_fmt_pct(winner['positive_active_ratio'])}</b></div>
              </div>
              {f'<p class="custom-note">自研网络：宏 F1 {_fmt_pct(custom["macro_f1"])}，代理收益 {_fmt_pct(custom["total_return_proxy"])}，活跃比例 {_fmt_pct(custom["active_ratio"])}。</p>' if custom is not None else ''}
            </article>
            """
        )

    top_rows = []
    for idx, row in ranked.head(16).iterrows():
        family_class = "custom" if row["family"] == "自研神经网络" else "baseline" if row["family"] == "基准" else "legacy"
        top_rows.append(
            f"""
            <tr class="{family_class}">
              <td>{idx + 1}</td>
              <td>{_html_escape(row['tenor'])}</td>
              <td>{_html_escape(row['model'])}</td>
              <td>{_html_escape(row['family'])}</td>
              <td>{_fmt_num(row['balanced_score'])}</td>
              <td>{_bar(row['macro_f1'], '#425f9c')}{_fmt_pct(row['macro_f1'])}</td>
              <td>{_fmt_pct(row['accuracy'])}</td>
              <td>{_fmt_pct(row['total_return_proxy'])}</td>
              <td>{_fmt_pct(row['active_ratio'])}</td>
              <td>{_fmt_pct(row['positive_active_ratio'])}</td>
              <td>{_fmt_pct(row['avg_confidence'])}</td>
            </tr>
            """
        )

    detail_sections = []
    for tenor, group in summary.groupby("tenor", sort=False):
        rows = []
        for _, row in group.sort_values(["balanced_score", "macro_f1"], ascending=False).iterrows():
            rows.append(
                f"""
                <tr>
                  <td>{_html_escape(row['model'])}</td>
                  <td>{_html_escape(row['family'])}</td>
                  <td>{_fmt_num(row['balanced_score'])}</td>
                  <td>{_fmt_pct(row['macro_f1'])}</td>
                  <td>{_fmt_pct(row['bearish_f1'])}</td>
                  <td>{_fmt_pct(row['bullish_f1'])}</td>
                  <td>{_fmt_pct(row['range_f1'])}</td>
                  <td>{_fmt_pct(row['total_return_proxy'])}</td>
                  <td>{_fmt_pct(row['max_drawdown_proxy'])}</td>
                  <td>{_fmt_pct(row['active_ratio'])}</td>
                  <td>{_fmt_pct(row['overconfident_error_ratio'])}</td>
                </tr>
                """
            )
        detail_sections.append(
            f"""
            <section class="section">
              <h2>{_html_escape(tenor)} 期限明细</h2>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>模型</th><th>类型</th><th>综合分</th><th>宏 F1</th><th>看空 F1</th><th>看多 F1</th><th>震荡 F1</th><th>代理收益</th><th>最大回撤</th><th>活跃比例</th><th>高置信错判</th></tr></thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </section>
            """
        )

    diag_html = "".join(f"<li>{_html_escape(note)}</li>" for note in diagnostics)
    conclusion = (
        "自研 CreditCurveNet 目前还不能替代旧模型。它的主要问题不是算力不够，而是样本外阶段过度激进、"
        "震荡识别不足、部分期限概率过度自信。下一步应该做概率校准、震荡保护层、多种随机种子集成，"
        "再进入滚动重训竞技场。"
    )
    if custom_avg_f1 > old_avg_f1:
        conclusion = "自研 CreditCurveNet 的平均宏 F1 已经接近或超过旧模型，但仍需要滚动重训确认稳定性，暂不直接替换日报。"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 模型竞技场</title>
  <style>
    :root {{
      --ink:#17202c; --muted:#5d6b7c; --line:#dfe6ee; --soft:#f5f7fa; --panel:#ffffff;
      --red:#d94848; --green:#1f8a62; --gold:#bf8427; --blue:#425f9c;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:#eef2f6; }}
    .wrap {{ width:min(1180px, calc(100% - 36px)); margin:0 auto; }}
    .hero {{ padding:34px 0 24px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#fbfcfd,#eef2f6); }}
    .eyebrow {{ color:var(--blue); font-weight:700; letter-spacing:.02em; }}
    h1 {{ margin:8px 0 10px; font-size:38px; line-height:1.12; }}
    .hero-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:22px; align-items:stretch; }}
    .hero p {{ margin:0; color:var(--muted); font-size:16px; line-height:1.7; }}
    .score-box {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 10px 28px rgba(35,48,66,.08); }}
    .score-box span {{ color:var(--muted); font-size:13px; }}
    .score-box strong {{ display:block; margin-top:7px; font-size:28px; color:var(--red); }}
    .main {{ padding:20px 0 42px; display:grid; gap:16px; }}
    .section, .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 22px rgba(35,48,66,.05); }}
    .section {{ padding:20px; }}
    h2 {{ margin:0 0 14px; font-size:22px; }}
    .takeaway {{ font-size:17px; line-height:1.75; color:#253140; margin:0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:16px; }}
    .metrics div {{ background:var(--soft); border-top:3px solid var(--blue); border-radius:6px; padding:10px 12px; }}
    .metrics span {{ display:block; color:var(--muted); font-size:12px; }}
    .metrics b {{ display:block; margin-top:5px; font-size:18px; }}
    .card-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ padding:16px; }}
    .card-head {{ display:flex; justify-content:space-between; gap:8px; align-items:start; }}
    .card-head span {{ font-size:22px; font-weight:800; }}
    .card-head strong {{ color:var(--red); text-align:right; }}
    .muted, .custom-note {{ color:var(--muted); line-height:1.55; }}
    .custom-note {{ margin:12px 0 0; font-size:13px; }}
    .mini {{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .mini div {{ padding:8px; }}
    .mini b {{ font-size:15px; }}
    .diagnosis {{ margin:0; padding-left:22px; line-height:1.8; color:#263241; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:930px; background:#fff; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid #edf1f5; text-align:left; font-size:13px; white-space:nowrap; }}
    th {{ color:#526173; background:#f7f9fb; font-weight:700; }}
    tr.custom td:first-child, tr.custom td:nth-child(3) {{ color:var(--red); font-weight:700; }}
    tr.baseline td:first-child {{ color:var(--gold); }}
    .bar {{ display:inline-block; vertical-align:middle; width:76px; height:8px; background:#e8edf2; border-radius:99px; margin-right:7px; overflow:hidden; }}
    .bar i {{ display:block; height:100%; border-radius:99px; }}
    .files {{ margin:0; color:var(--muted); line-height:1.75; }}
    .files code {{ color:#21314a; background:#f1f4f7; padding:2px 5px; border-radius:4px; }}
    @media (max-width:900px) {{
      .hero-grid,.card-grid {{ grid-template-columns:1fr; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      h1 {{ font-size:30px; }}
    }}
    @media print {{
      @page {{ size:A4; margin:8mm; }}
      body {{ background:#fff; }}
      .wrap {{ width:100%; }}
      .hero {{ padding:10px 0 12px; }}
      h1 {{ font-size:26px; }}
      .section,.card {{ box-shadow:none; break-inside:avoid; }}
      .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .table-wrap {{ overflow:visible; }}
      th,td {{ font-size:10px; padding:6px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap hero-grid">
      <div>
        <div class="eyebrow">信用债 AI 研究辅助 · 模型竞技场</div>
        <h1>旧模型、自研网络与基准策略统一评分</h1>
        <p>所有模型按相同日期集合重新评分，区间为 {date_start} 至 {date_end}。指标只用于研究和决策辅助，不构成自动交易指令。收益为久期方向代理收益，不含交易成本和真实组合约束。</p>
      </div>
      <aside class="score-box">
        <span>当前综合冠军</span>
        <strong>{_html_escape(best['tenor'])} · {_html_escape(best['model'])}</strong>
        <p class="muted">综合分 {_fmt_num(best['balanced_score'])}；宏 F1 {_fmt_pct(best['macro_f1'])}；代理收益 {_fmt_pct(best['total_return_proxy'])}。</p>
      </aside>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{_html_escape(conclusion)}</p>
      <div class="metrics">
        <div><span>统一评分模型数</span><b>{n_models}</b></div>
        <div><span>旧模型平均宏 F1</span><b>{_fmt_pct(old_avg_f1)}</b></div>
        <div><span>自研网络平均宏 F1</span><b>{_fmt_pct(custom_avg_f1)}</b></div>
        <div><span>评分区间</span><b>{date_start} 至 {date_end}</b></div>
      </div>
    </section>
    <section>
      <div class="card-grid">{''.join(cards)}</div>
    </section>
    <section class="section">
      <h2>总排行榜</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>名次</th><th>期限</th><th>模型</th><th>类型</th><th>综合分</th><th>宏 F1</th><th>准确率</th><th>代理收益</th><th>活跃比例</th><th>活跃胜率</th><th>平均置信度</th></tr></thead>
          <tbody>{''.join(top_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>自研网络诊断</h2>
      <ul class="diagnosis">{diag_html}</ul>
    </section>
    {''.join(detail_sections)}
    <section class="section">
      <h2>口径和文件</h2>
      <p class="files">数据来源：DM 曲线和宏观指标清洗后的宽表。评价方式：读取各模型 <code>test_predictions.csv</code>，在同一期限所有模型共有日期上重新计算准确率、宏 F1、看空/看多/震荡 F1、代理收益、回撤、信号活跃比例和高置信错判比例。综合分不是投资收益，它只是为了排序展示。</p>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def run_model_arena(
    features_path: str | Path = "data/dm_daily_master_curve_2020/processed/dm_features_latest.csv",
    models_root: str | Path = "models",
    out_dir: str | Path = "data/model_arena/curve_2020_v1",
) -> dict[str, Path]:
    features_path = Path(features_path)
    models_root = Path(models_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _load_features(features_path)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []

    for tenor in TENORS:
        raw_candidates = _load_model_candidates(models_root, tenor)
        if not raw_candidates:
            continue
        dates = _common_dates(raw_candidates)
        if not dates:
            continue
        aligned_candidates = []
        base = _align_candidate(raw_candidates[0], dates)
        base = base[["date", "y_true", "future_yield_change"]].copy()
        for candidate in raw_candidates:
            aligned = _align_candidate(candidate, dates)
            aligned_candidates.append({**candidate, "df": aligned})

        legacy_aligned = [item["df"] for item in aligned_candidates if item["family"] == "旧模型"]
        all_aligned = [item["df"] for item in aligned_candidates]
        old_ensemble = _build_ensemble("旧模型-概率均值集成", "旧模型集成", legacy_aligned, base)
        all_ensemble = _build_ensemble("新旧-概率均值集成", "混合集成", all_aligned, base)
        for maybe in (old_ensemble, all_ensemble):
            if maybe is not None:
                aligned_candidates.append(maybe)
        aligned_candidates.extend(_baseline_predictions(features, tenor, base))

        for candidate in aligned_candidates:
            row, detail = _score_predictions(
                df=candidate["df"],
                tenor=tenor,
                model=candidate["model"],
                family=candidate["family"],
                source_path=str(candidate.get("path", "")),
            )
            summary_rows.append(row)
            detail_rows.append(detail)

    if not summary_rows:
        raise ValueError("没有找到可评分的模型预测文件。")

    summary = pd.DataFrame(summary_rows).sort_values(["tenor", "balanced_score"], ascending=[True, False])
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    diagnostics = _diagnose_custom(summary)

    summary_path = out_dir / "arena_summary.csv"
    details_path = out_dir / "arena_predictions.csv"
    diagnostics_path = out_dir / "arena_diagnostics.json"
    html_path = out_dir / "model_arena_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    diagnostics_path.write_text(json.dumps({"diagnostics": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(summary, details, diagnostics, html_path)
    return {
        "summary": summary_path,
        "predictions": details_path,
        "diagnostics": diagnostics_path,
        "html": html_path,
    }
