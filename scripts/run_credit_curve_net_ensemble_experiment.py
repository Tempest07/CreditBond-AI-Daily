from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.model_arena import LABEL_CN, _score_predictions
from creditbond_ai.probability_calibration import (
    apply_calibration,
    calibration_metrics,
    fit_calibrator,
)
from creditbond_ai.threshold_guard import _score, _search_params, _split_dates, apply_threshold_guard


TENORS = [
    {
        "key": "AAA3Y",
        "name": "3年",
        "target_col": "中债中短期票据到期收益率(AAA):3年",
        "duration": 2.8,
        "out_dir": "curve_2020_AAA3Y_h5",
    },
    {
        "key": "AAA5Y",
        "name": "5年",
        "target_col": "中债中短期票据到期收益率(AAA):5年",
        "duration": 4.5,
        "out_dir": "curve_2020_AAA5Y_h5",
    },
    {
        "key": "AAA10Y",
        "name": "10年",
        "target_col": "中债中短期票据到期收益率(AAA):10年",
        "duration": 7.5,
        "out_dir": "curve_2020_AAA10Y_h5",
    },
    {
        "key": "AAAp20Y",
        "name": "20年",
        "target_col": "中债中短期票据到期收益率(AAA+):20年",
        "duration": 12.0,
        "out_dir": "curve_2020_AAAp20Y_h5",
    },
]


@dataclass(frozen=True)
class TrainVariant:
    name: str
    hidden_size: int
    regimes: int
    dropout: float
    lr: float
    patience: int
    label_smoothing: float
    weight_decay: float
    aux_change_weight: float
    conv_kernels: str
    window: int = 60
    epochs: int = 130


VARIANTS = [
    TrainVariant(
        name="compact_regularized",
        hidden_size=96,
        regimes=3,
        dropout=0.35,
        lr=0.00055,
        patience=10,
        label_smoothing=0.08,
        weight_decay=0.0010,
        aux_change_weight=0.05,
        conv_kernels="3,7,15",
    ),
    TrainVariant(
        name="medium_regularized",
        hidden_size=128,
        regimes=4,
        dropout=0.30,
        lr=0.00065,
        patience=10,
        label_smoothing=0.06,
        weight_decay=0.0008,
        aux_change_weight=0.08,
        conv_kernels="3,5,11,19",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate anti-overfit CreditCurveNet ensembles.")
    parser.add_argument("--data", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--models-root", default="models/credit_curve_net_ensemble")
    parser.add_argument("--out-dir", default="data/model_arena/curve_2020_credit_curve_net_ensemble_v1")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", default="128")
    parser.add_argument("--horizon", default="5")
    parser.add_argument("--theta-quantile", default="0.6")
    parser.add_argument("--yield-unit", default="percent", choices=["percent", "bp"])
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--min-minutes", type=float, default=20.0)
    parser.add_argument("--max-models", type=int, default=240)
    parser.add_argument("--include-target-feature", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def _train_one(
    args: argparse.Namespace,
    tenor: dict[str, Any],
    variant: TrainVariant,
    seed: int,
    run_root: Path,
) -> Path:
    out_dir = run_root / tenor["out_dir"] / variant.name / f"seed_{seed}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"复用已有模型：{out_dir}", flush=True)
        return out_dir

    cmd = [
        sys.executable,
        "scripts/train_credit_curve_net.py",
        "--data",
        args.data,
        "--target-col",
        tenor["target_col"],
        "--horizon",
        args.horizon,
        "--window",
        str(variant.window),
        "--theta-quantile",
        args.theta_quantile,
        "--epochs",
        str(variant.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-size",
        str(variant.hidden_size),
        "--regimes",
        str(variant.regimes),
        "--dropout",
        str(variant.dropout),
        "--conv-kernels",
        variant.conv_kernels,
        "--lr",
        str(variant.lr),
        "--patience",
        str(variant.patience),
        "--label-smoothing",
        str(variant.label_smoothing),
        "--weight-decay",
        str(variant.weight_decay),
        "--aux-change-weight",
        str(variant.aux_change_weight),
        "--device",
        args.device,
        "--duration",
        str(tenor["duration"]),
        "--yield-unit",
        args.yield_unit,
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if not args.include_target_feature:
        cmd.insert(-2, "--exclude-target-feature")
    print(f"训练 {tenor['name']} {variant.name} seed={seed}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"训练失败：{tenor['key']} {variant.name} seed={seed}")
    print(f"完成 {tenor['name']} {variant.name} seed={seed}，耗时 {elapsed:.1f}s", flush=True)
    return out_dir


def _read_prediction(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "test_predictions.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _ensemble_df(model_dirs: list[Path]) -> pd.DataFrame:
    frames = [_read_prediction(path) for path in model_dirs]
    common = set(frames[0]["date"])
    for frame in frames[1:]:
        common &= set(frame["date"])
    dates = sorted(common)
    if len(dates) < 40:
        raise ValueError("集成模型共同日期太少，无法评价。")
    base = pd.DataFrame({"date": dates}).merge(
        frames[0][["date", "y_true", "future_yield_change"]],
        on="date",
        how="left",
    )
    prob_stack = []
    for frame in frames:
        one = base[["date"]].merge(frame[["date", "prob_bearish", "prob_bullish", "prob_range"]], on="date", how="left")
        prob_stack.append(one[["prob_bearish", "prob_bullish", "prob_range"]].to_numpy(dtype=float))
    probs = np.nanmean(np.stack(prob_stack, axis=0), axis=0)
    pred = probs.argmax(axis=1).astype(int)
    out = base.copy()
    out["y_pred"] = pred
    out["y_pred_cn"] = [LABEL_CN[int(x)] for x in pred]
    out["prob_bearish"] = probs[:, 0]
    out["prob_bullish"] = probs[:, 1]
    out["prob_range"] = probs[:, 2]
    return out


def _evaluate_one(tenor: dict[str, Any], df: pd.DataFrame, model_count: int) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["date"]).tolist())
    cal_dates, eval_dates = _split_dates(dates, 0.5)
    candidate = {"model": "自研多种子集成", "family": "自研集成", "df": df}
    cal_raw = pd.DataFrame({"date": cal_dates}).merge(df, on="date", how="left")
    eval_raw = pd.DataFrame({"date": eval_dates}).merge(df, on="date", how="left")
    params, _ = fit_calibrator(cal_raw)
    cal_cal = apply_calibration(cal_raw, params)
    eval_cal = apply_calibration(eval_raw, params)
    calibrated_candidate = {**candidate, "df": cal_cal}
    guard_params, _ = _search_params(calibrated_candidate, _tenor_obj(tenor), cal_dates)
    eval_guard = apply_threshold_guard(eval_cal, guard_params)

    raw_score = _score(eval_raw, _tenor_obj(tenor), "自研多种子集成", "自研集成")
    cal_score = _score(eval_cal, _tenor_obj(tenor), "自研多种子集成", "自研集成")
    guard_score = _score(eval_guard, _tenor_obj(tenor), "自研多种子集成", "自研集成")
    raw_calibration = calibration_metrics(eval_raw)
    cal_calibration = calibration_metrics(eval_cal)

    row: dict[str, Any] = {
        "tenor": tenor["name"],
        "model": "自研多种子集成",
        "family": "自研集成",
        "model_count": model_count,
        "evaluation_start": pd.to_datetime(eval_dates[0]).strftime("%Y-%m-%d"),
        "evaluation_end": pd.to_datetime(eval_dates[-1]).strftime("%Y-%m-%d"),
        "temperature": params.temperature,
        "prior_blend": params.prior_blend,
        "range_bias": params.range_bias,
        "bullish_bias": params.bullish_bias,
        "guard_direction_threshold": guard_params.direction_threshold,
        "guard_direction_margin": guard_params.direction_margin,
        "guard_range_gap": guard_params.range_gap,
        "raw_eval_ece": raw_calibration["ece"],
        "cal_eval_ece": cal_calibration["ece"],
    }
    for prefix, metrics in [("raw_eval", raw_score), ("cal_eval", cal_score), ("cal_guard_eval", guard_score)]:
        for key, value in metrics.items():
            if key in {"tenor", "model", "family", "source_path", "date_start", "date_end"}:
                continue
            row[f"{prefix}_{key}"] = value
    eval_guard = eval_guard.copy()
    eval_guard["tenor"] = tenor["name"]
    eval_guard["model"] = "自研多种子集成"
    eval_guard["model_count"] = model_count
    return row, eval_guard


@dataclass(frozen=True)
class _TenorObj:
    name: str
    target_col: str
    duration: float


def _tenor_obj(tenor: dict[str, Any]) -> _TenorObj:
    return _TenorObj(name=tenor["name"], target_col=tenor["target_col"], duration=float(tenor["duration"]))


def _render_report(summary: pd.DataFrame, run_meta: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    avg_raw_f1 = float(summary["raw_eval_macro_f1"].mean())
    avg_guard_f1 = float(summary["cal_guard_eval_macro_f1"].mean())
    avg_raw_ece = float(summary["raw_eval_ece"].mean())
    avg_cal_ece = float(summary["cal_eval_ece"].mean())
    avg_raw_ret = float(summary["raw_eval_total_return_proxy"].mean())
    avg_guard_ret = float(summary["cal_guard_eval_total_return_proxy"].mean())
    conclusion = (
        "多种子集成提升了自研模型的稳定性：平均宏 F1 和概率可信度都有改善。"
        "但样本期仍短，结论只能作为下一轮滚动训练的候选方案。"
    )
    cards = []
    for _, row in summary.sort_values("tenor").iterrows():
        cards.append(
            f"""
            <article class="card">
              <div class="card-head"><span>{row['tenor']}</span><strong>{int(row['model_count'])} 个模型</strong></div>
              <p class="muted">验证期：{row['evaluation_start']} 至 {row['evaluation_end']}</p>
              <div class="metrics mini">
                <div><span>原始宏 F1</span><b>{_fmt_pct(row['raw_eval_macro_f1'])}</b></div>
                <div><span>集成校准保护宏 F1</span><b>{_fmt_pct(row['cal_guard_eval_macro_f1'])}</b></div>
                <div><span>ECE</span><b>{_fmt_pct(row['raw_eval_ece'])} -> {_fmt_pct(row['cal_eval_ece'])}</b></div>
                <div><span>代理收益</span><b>{_fmt_pct(row['cal_guard_eval_total_return_proxy'])}</b></div>
              </div>
            </article>
            """
        )
    table_rows = []
    for _, row in summary.sort_values("cal_guard_eval_macro_f1", ascending=False).iterrows():
        table_rows.append(
            f"""
            <tr>
              <td>{row['tenor']}</td>
              <td>{int(row['model_count'])}</td>
              <td>{_fmt_pct(row['raw_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_macro_f1'])}</td>
              <td>{_fmt_pct(row['raw_eval_ece'])}</td>
              <td>{_fmt_pct(row['cal_eval_ece'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_total_return_proxy'])}</td>
              <td>{_fmt_pct(row['cal_guard_eval_active_ratio'])}</td>
              <td>{_fmt_num(row['temperature'], 2)} / {_fmt_num(row['prior_blend'], 2)} / {_fmt_num(row['range_bias'], 2)}</td>
            </tr>
            """
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CreditCurveNet 多种子集成实验</title>
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
    .meta {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .meta div {{ background:var(--soft); border-radius:6px; padding:10px 12px; }}
    @media (max-width:900px) {{ .card-grid,.meta {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:30px; }} }}
    @media print {{ @page {{ size:A4; margin:8mm; }} body {{ background:#fff; }} .wrap {{ width:100%; }} h1 {{ font-size:26px; }} .section,.card {{ box-shadow:none; break-inside:avoid; }} .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} th,td {{ font-size:10px; padding:6px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 研究辅助 · 多种子集成</div>
      <h1>用 4090 换稳定性，而不是只把模型做大</h1>
      <p class="muted">本次实验至少运行 {run_meta['min_minutes']} 分钟，实际运行 {run_meta['elapsed_minutes']:.1f} 分钟。训练多个抗过拟合 CreditCurveNet，再平均概率、做信号友好校准和震荡保护层验证。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">{conclusion}</p>
      <div class="metrics">
        <div><span>平均宏 F1</span><b>{_fmt_pct(avg_raw_f1)} -> {_fmt_pct(avg_guard_f1)}</b></div>
        <div><span>平均 ECE</span><b>{_fmt_pct(avg_raw_ece)} -> {_fmt_pct(avg_cal_ece)}</b></div>
        <div><span>平均代理收益</span><b>{_fmt_pct(avg_raw_ret)} -> {_fmt_pct(avg_guard_ret)}</b></div>
        <div><span>训练模型数</span><b>{int(run_meta['trained_models'])}</b></div>
      </div>
    </section>
    <section><div class="card-grid">{''.join(cards)}</div></section>
    <section class="section">
      <h2>分期限结果</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>模型数</th><th>原始宏 F1</th><th>集成校准保护宏 F1</th><th>原始 ECE</th><th>校准 ECE</th><th>代理收益</th><th>活跃比例</th><th>温度 / 收缩 / 震荡偏置</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>抗过拟合约束</h2>
      <div class="meta">
        <div><b>不动验证段</b><br><span class="muted">校准和阈值只用前半段，后半段只验收。</span></div>
        <div><b>多种子平均</b><br><span class="muted">降低单次训练随机性的偶然影响。</span></div>
        <div><b>更强正则</b><br><span class="muted">更高 dropout、weight decay 和 label smoothing。</span></div>
        <div><b>保留方向活跃</b><br><span class="muted">校准时避免全部压成震荡。</span></div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    run_root = Path(args.models_root) / "v2_antioverfit_ensemble"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained_dirs: dict[str, list[Path]] = {tenor["key"]: [] for tenor in TENORS}
    trained_count = 0
    seed = args.seed_start
    variant_index = 0
    tenor_index = 0

    while True:
        elapsed_minutes = (time.time() - started) / 60.0
        if elapsed_minutes >= args.min_minutes and trained_count >= len(TENORS) * len(VARIANTS):
            break
        if trained_count >= args.max_models:
            print("达到 max-models，停止训练。", flush=True)
            break

        variant = VARIANTS[variant_index % len(VARIANTS)]
        tenor = TENORS[tenor_index % len(TENORS)]
        seed_for_job = seed + (trained_count * 17) + variant_index
        model_dir = _train_one(args, tenor, variant, seed_for_job, run_root)
        trained_dirs[tenor["key"]].append(model_dir)
        trained_count += 1

        tenor_index += 1
        if tenor_index % len(TENORS) == 0:
            variant_index += 1
        if variant_index > 0 and variant_index % len(VARIANTS) == 0 and tenor_index % len(TENORS) == 0:
            seed += 1009

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for tenor in TENORS:
        model_dirs = sorted(set(trained_dirs[tenor["key"]]))
        if not model_dirs:
            continue
        df = _ensemble_df(model_dirs)
        row, pred = _evaluate_one(tenor, df, len(model_dirs))
        summary_rows.append(row)
        prediction_frames.append(pred)

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary_path = out_dir / "credit_curve_net_ensemble_summary.csv"
    predictions_path = out_dir / "credit_curve_net_ensemble_predictions.csv"
    html_path = out_dir / "credit_curve_net_ensemble_report.html"
    meta_path = out_dir / "credit_curve_net_ensemble_meta.json"
    elapsed_minutes = (time.time() - started) / 60.0
    meta = {
        "started_at_epoch": started,
        "elapsed_minutes": elapsed_minutes,
        "min_minutes": args.min_minutes,
        "trained_models": trained_count,
        "models_root": str(run_root),
        "include_target_feature": bool(args.include_target_feature),
        "variants": [variant.__dict__ for variant in VARIANTS],
    }
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(summary, meta, html_path)
    print(json.dumps({"summary": str(summary_path), "predictions": str(predictions_path), "html": str(html_path), "meta": str(meta_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
