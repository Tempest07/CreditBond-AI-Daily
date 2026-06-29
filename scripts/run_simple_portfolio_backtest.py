from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.position_backtest import (
    _annualized_return,
    _infer_duration,
    _load_checkpoint,
    _max_drawdown,
    _predict_all_history,
    _prepare_asset_returns,
    _sharpe,
    parse_duration_map,
)


DEFAULT_MODEL_DIRS = [
    "models/curve_2020_AAA3Y_h5/01_full_features/gru",
    "models/curve_2020_AAA3Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA3Y_h5/01_full_features/transformer",
    "models/curve_2020_AAA5Y_h5/01_full_features/gru",
    "models/curve_2020_AAA5Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA5Y_h5/01_full_features/transformer",
    "models/curve_2020_AAA10Y_h5/01_full_features/gru",
    "models/curve_2020_AAA10Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA10Y_h5/01_full_features/transformer",
    "models/curve_2020_AAAp20Y_h5/01_full_features/gru",
    "models/curve_2020_AAAp20Y_h5/01_full_features/tcn",
    "models/curve_2020_AAAp20Y_h5/01_full_features/transformer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple four-tenor ensemble portfolio backtest.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--out-dir", default="data/backtests/curve_2020_simple_portfolio")
    parser.add_argument("--model-dir", action="append", default=[])
    parser.add_argument("--rolling-summary", default="")
    parser.add_argument("--initial-position", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--transaction-cost-bp", type=float, default=0.5)
    parser.add_argument("--prob-threshold", type=float, default=0.45)
    parser.add_argument("--margin-vs-range", type=float, default=0.03)
    parser.add_argument("--annual-days", type=int, default=252)
    parser.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    parser.add_argument("--no-carry", action="store_true")
    return parser.parse_args()


def clamp_position(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def infer_tenor(target_col: str) -> str:
    for tenor in ["20", "10", "5", "3"]:
        if f":{tenor}年" in target_col:
            return tenor
    return ""


def tenor_label(target_col: str) -> str:
    tenor = infer_tenor(target_col)
    rating = "AAA+" if "AAA+" in target_col else "AAA"
    return f"{rating} {tenor}年" if tenor else target_col


def load_rolling_ensemble_predictions(rolling_summary_path: str | Path) -> dict[str, dict[str, Any]]:
    summary = pd.read_csv(rolling_summary_path, encoding="utf-8-sig")
    required = {"fold", "target_col", "model_dir"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"滚动验证汇总缺少字段：{sorted(missing)}")

    ensembles: dict[str, dict[str, Any]] = {}
    checkpoint_by_target: dict[str, Any] = {}
    model_dirs_by_target: dict[str, list[str]] = {}
    pred_parts_by_target: dict[str, list[pd.DataFrame]] = {}

    for (fold, target_col), group in summary.groupby(["fold", "target_col"], sort=True):
        frames = []
        model_dirs = []
        for idx, row in group.reset_index(drop=True).iterrows():
            model_dir = Path(str(row["model_dir"]))
            checkpoint = _load_checkpoint(model_dir)
            checkpoint_by_target.setdefault(str(target_col), checkpoint)
            model_dirs.append(str(model_dir))
            pred_path = model_dir / "test_predictions.csv"
            if not pred_path.exists():
                raise FileNotFoundError(f"找不到滚动测试预测文件：{pred_path}")
            pred = pd.read_csv(pred_path, encoding="utf-8-sig")
            pred["date"] = pd.to_datetime(pred["date"])
            pred = pred.set_index("date")[["prob_bearish", "prob_bullish", "prob_range"]]
            frames.append(pred.add_suffix(f"__m{idx}"))

        merged = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        if merged.empty:
            continue
        out = pd.DataFrame(index=merged.index)
        for col in ["prob_bearish", "prob_bullish", "prob_range"]:
            cols = [name for name in merged.columns if name.startswith(f"{col}__m")]
            out[col] = merged[cols].mean(axis=1)
        out = out.reset_index().sort_values("date")
        out["fold"] = int(fold)
        pred_parts_by_target.setdefault(str(target_col), []).append(out)
        model_dirs_by_target.setdefault(str(target_col), []).extend(model_dirs)

    for target_col, parts in pred_parts_by_target.items():
        pred = pd.concat(parts, ignore_index=True)
        pred = pred.sort_values(["date", "fold"]).drop_duplicates("date", keep="last").reset_index(drop=True)
        ensembles[target_col] = {
            "target_col": target_col,
            "display_name": tenor_label(target_col),
            "model_dirs": sorted(set(model_dirs_by_target.get(target_col, []))),
            "checkpoint": checkpoint_by_target[target_col],
            "pred": pred,
        }
    return ensembles


def load_ensemble_predictions(model_dirs: list[str], features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for model_dir_raw in model_dirs:
        model_dir = Path(model_dir_raw)
        checkpoint = _load_checkpoint(model_dir)
        target_col = str(checkpoint.get("target_col", ""))
        if not target_col:
            raise ValueError(f"模型缺少 target_col: {model_dir}")
        pred = _predict_all_history(model_dir, checkpoint, features)
        grouped.setdefault(target_col, []).append(
            {
                "model_dir": str(model_dir),
                "checkpoint": checkpoint,
                "pred": pred,
            }
        )

    ensembles: dict[str, dict[str, Any]] = {}
    for target_col, items in grouped.items():
        frames = []
        for idx, item in enumerate(items):
            pred = item["pred"].copy()
            pred["date"] = pd.to_datetime(pred["date"])
            pred = pred.set_index("date")[["prob_bearish", "prob_bullish", "prob_range"]]
            frames.append(pred.add_suffix(f"__m{idx}"))
        merged = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        out = pd.DataFrame(index=merged.index)
        for col in ["prob_bearish", "prob_bullish", "prob_range"]:
            cols = [name for name in merged.columns if name.startswith(f"{col}__m")]
            out[col] = merged[cols].mean(axis=1)
        out = out.reset_index().sort_values("date")
        ensembles[target_col] = {
            "target_col": target_col,
            "display_name": tenor_label(target_col),
            "model_dirs": [item["model_dir"] for item in items],
            "checkpoint": items[0]["checkpoint"],
            "pred": out,
        }
    return ensembles


def apply_rule(pred: pd.DataFrame, prob_threshold: float, margin_vs_range: float) -> pd.DataFrame:
    out = pred.copy()
    bear = out["prob_bearish"].astype(float)
    bull = out["prob_bullish"].astype(float)
    range_prob = out["prob_range"].astype(float)
    bullish_side = bull >= bear
    direction_prob = np.where(bullish_side, bull, bear)
    direction_signal = np.where(bullish_side, 1, -1)
    active = (direction_prob >= prob_threshold) & ((direction_prob - range_prob) >= margin_vs_range)
    out["signal"] = np.where(active, direction_signal, 0).astype(int)
    out["prediction"] = np.select(
        [out["signal"].eq(-1), out["signal"].eq(1)],
        ["看空", "看多"],
        default="震荡",
    )
    out["direction_prob"] = direction_prob
    return out


def simulate_tenor(
    entity: dict[str, Any],
    features: pd.DataFrame,
    duration_map: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_col = entity["target_col"]
    duration = _infer_duration(target_col, entity["checkpoint"], duration_map)
    pred = apply_rule(entity["pred"], args.prob_threshold, args.margin_vs_range)
    asset = _prepare_asset_returns(
        features=features,
        target_col=target_col,
        duration=duration,
        annual_days=args.annual_days,
        include_carry=not args.no_carry,
    )
    merged = pred.merge(asset, on="date", how="inner").sort_values("date")
    if merged.empty:
        raise ValueError(f"{entity['display_name']} 没有可回测日期")

    position = clamp_position(args.initial_position)
    value = 1.0
    buy_hold_value = 1.0
    neutral_value = 1.0
    rows = []
    for _, row in merged.iterrows():
        signal = int(row["signal"])
        before = position
        if signal > 0:
            position = clamp_position(position + args.step)
        elif signal < 0:
            position = clamp_position(position - args.step)
        trade = abs(position - before)
        cost = trade * (args.transaction_cost_bp / 10000.0)
        bond_return = float(row["bond_return_proxy"])
        strategy_return = position * bond_return - cost
        buy_hold_return = bond_return
        neutral_return = args.initial_position * bond_return
        value *= 1.0 + strategy_return
        buy_hold_value *= 1.0 + buy_hold_return
        neutral_value *= 1.0 + neutral_return
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "next_date": pd.Timestamp(row["next_date"]).date().isoformat(),
                "tenor": entity["display_name"],
                "prediction": row["prediction"],
                "signal": signal,
                "prob_bearish": float(row["prob_bearish"]),
                "prob_bullish": float(row["prob_bullish"]),
                "prob_range": float(row["prob_range"]),
                "position_before": before,
                "position_after": position,
                "trade": trade,
                "yield": float(row[target_col]),
                "next_yield": float(row["next_yield"]),
                "yield_change_1d": float(row["yield_change_1d"]),
                "bond_return": bond_return,
                "strategy_return": strategy_return,
                "buy_hold_return": buy_hold_return,
                "neutral_return": neutral_return,
                "strategy_value": value,
                "buy_hold_value": buy_hold_value,
                "neutral_value": neutral_value,
            }
        )
    detail = pd.DataFrame(rows)
    periods = len(detail)
    metrics = {
        "tenor": entity["display_name"],
        "target_col": target_col,
        "model_count": len(entity["model_dirs"]),
        "start_date": str(detail["date"].iloc[0]),
        "end_date": str(detail["next_date"].iloc[-1]),
        "periods": periods,
        "duration": duration,
        "total_return": float(detail["strategy_value"].iloc[-1] - 1.0),
        "annualized_return": _annualized_return(float(detail["strategy_value"].iloc[-1] - 1.0), periods, args.annual_days),
        "max_drawdown": _max_drawdown(detail["strategy_value"]),
        "positive_day_ratio": float((detail["strategy_return"] > 0).mean()),
        "final_position": float(detail["position_after"].iloc[-1]),
        "average_position": float(detail["position_after"].mean()),
        "turnover": float(detail["trade"].sum()),
        "trade_count": int((detail["trade"] > 0).sum()),
        "signal_counts": {str(k): int(v) for k, v in detail["prediction"].value_counts().to_dict().items()},
        "buy_hold_total_return": float(detail["buy_hold_value"].iloc[-1] - 1.0),
        "neutral_total_return": float(detail["neutral_value"].iloc[-1] - 1.0),
    }
    return metrics, detail


def portfolio_from_tenors(details: list[pd.DataFrame], annual_days: int) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = []
    for detail in details:
        cols = ["date", "next_date", "tenor", "strategy_return", "buy_hold_return", "neutral_return", "position_after", "trade"]
        frames.append(detail[cols].copy())
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby(["date", "next_date"], as_index=False).agg(
        strategy_return=("strategy_return", "mean"),
        buy_hold_return=("buy_hold_return", "mean"),
        neutral_return=("neutral_return", "mean"),
        average_position=("position_after", "mean"),
        daily_turnover=("trade", "sum"),
    )
    grouped = grouped.sort_values("date").reset_index(drop=True)
    grouped["strategy_value"] = (1.0 + grouped["strategy_return"]).cumprod()
    grouped["buy_hold_value"] = (1.0 + grouped["buy_hold_return"]).cumprod()
    grouped["neutral_value"] = (1.0 + grouped["neutral_return"]).cumprod()
    periods = len(grouped)
    total = float(grouped["strategy_value"].iloc[-1] - 1.0)
    buy_hold_total = float(grouped["buy_hold_value"].iloc[-1] - 1.0)
    neutral_total = float(grouped["neutral_value"].iloc[-1] - 1.0)
    metrics = {
        "start_date": str(grouped["date"].iloc[0]),
        "end_date": str(grouped["next_date"].iloc[-1]),
        "periods": periods,
        "total_return": total,
        "annualized_return": _annualized_return(total, periods, annual_days),
        "max_drawdown": _max_drawdown(grouped["strategy_value"]),
        "daily_volatility_annualized": float(grouped["strategy_return"].std(ddof=1) * math.sqrt(annual_days)) if periods > 1 else 0.0,
        "sharpe": _sharpe(grouped["strategy_return"], annual_days),
        "positive_day_ratio": float((grouped["strategy_return"] > 0).mean()),
        "average_position": float(grouped["average_position"].mean()),
        "turnover": float(grouped["daily_turnover"].sum()),
        "buy_hold_total_return": buy_hold_total,
        "buy_hold_annualized_return": _annualized_return(buy_hold_total, periods, annual_days),
        "buy_hold_max_drawdown": _max_drawdown(grouped["buy_hold_value"]),
        "neutral_total_return": neutral_total,
        "neutral_annualized_return": _annualized_return(neutral_total, periods, annual_days),
        "neutral_max_drawdown": _max_drawdown(grouped["neutral_value"]),
        "excess_vs_buy_hold": total - buy_hold_total,
        "excess_vs_neutral": total - neutral_total,
    }
    return metrics, grouped


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "不适用"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "不适用"
    return f"{float(value):.{digits}f}"


def svg_line_chart(rows: list[dict[str, Any]], width: int = 920, height: int = 300) -> str:
    if not rows:
        return ""
    keys = [
        ("strategy_value", "AI组合策略", "#2f6f9f"),
        ("buy_hold_value", "四期限买满基准", "#87919b"),
        ("neutral_value", "四期限50%静态仓位", "#b7812f"),
    ]
    pad_l, pad_r, pad_t, pad_b = 58, 24, 24, 42
    values = [float(row[key]) for row in rows for key, _, _ in keys]
    y_min, y_max = min(values), max(values)
    span = y_max - y_min or 0.02
    y_min -= span * 0.08
    y_max += span * 0.08

    def x_pos(i: int) -> float:
        return pad_l + i * (width - pad_l - pad_r) / max(1, len(rows) - 1)

    def y_pos(value: float) -> float:
        return pad_t + (y_max - value) * (height - pad_t - pad_b) / (y_max - y_min)

    grid = []
    for value in [y_min, (y_min + y_max) / 2, y_max]:
        y = y_pos(value)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#e4ebf0" />')
        grid.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#657180">{(value-1)*100:.1f}%</text>')
    lines = []
    legends = []
    for key, label, color in keys:
        coords = " ".join(f"{x_pos(i):.1f},{y_pos(float(row[key])):.1f}" for i, row in enumerate(rows))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />')
        legends.append(f'<span><i style="background:{color}"></i>{html.escape(label)}</span>')
    return f"""
    <div class="chart-box">
      <svg viewBox="0 0 {width} {height}">
        {''.join(grid)}
        {''.join(lines)}
        <text x="{pad_l}" y="{height-14}" font-size="12" fill="#657180">{html.escape(str(rows[0]["date"]))}</text>
        <text x="{width-pad_r}" y="{height-14}" text-anchor="end" font-size="12" fill="#657180">{html.escape(str(rows[-1]["next_date"]))}</text>
      </svg>
      <div class="legend">{''.join(legends)}</div>
    </div>
    """


def write_html(report: dict[str, Any], path: Path) -> None:
    portfolio = report["portfolio_metrics"]
    tenor_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row['tenor'])}</td>
          <td>{html.escape(row['start_date'])} 至 {html.escape(row['end_date'])}</td>
          <td>{fmt_pct(row['total_return'])}</td>
          <td>{fmt_pct(row['annualized_return'])}</td>
          <td>{fmt_pct(row['max_drawdown'])}</td>
          <td>{fmt_pct(row['buy_hold_total_return'])}</td>
          <td>{fmt_pct(row['final_position'], 1)}</td>
          <td>{fmt_num(row['turnover'], 2)}</td>
        </tr>
        """
        for row in report["tenor_metrics"]
    )
    config = report["config"]
    mode_title = "滚动样本外组合回测" if config.get("mode") == "rolling" else "简单组合回测"
    mode_note = (
        "本报告只使用每一折测试期预测来调仓，更接近真实事前决策。"
        if config.get("mode") == "rolling"
        else "本报告使用全历史回放信号，包含训练期，不等同于严格样本外业绩。"
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 简单组合回测</title>
  <style>
    :root {{
      --ink:#17202a; --muted:#64717f; --line:#dce4ea; --paper:#fff; --back:#f4f6f8;
      --blue:#2f6f9f; --green:#238a62; --red:#c74343; --amber:#b7812f;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    .wrap {{ width:min(1120px, calc(100% - 32px)); margin:0 auto; }}
    .top {{ background:var(--paper); border-bottom:1px solid var(--line); }}
    .hero {{ padding:34px 0 26px; display:grid; grid-template-columns:1.3fr .8fr; gap:24px; align-items:end; }}
    .eyebrow {{ color:var(--blue); font-weight:800; font-size:14px; margin-bottom:8px; }}
    h1 {{ margin:0; font-size:34px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    p {{ margin:10px 0 0; color:var(--muted); }}
    .box {{ border:1px solid var(--line); border-radius:8px; background:#fbfcfd; padding:16px; }}
    .box span {{ display:block; color:var(--muted); font-size:13px; }}
    .box strong {{ display:block; margin-top:4px; font-size:30px; color:var(--blue); }}
    .main {{ display:grid; gap:18px; padding:22px 0 46px; }}
    .section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:22px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .metric {{ background:#f8fafb; border-top:3px solid var(--blue); border-radius:6px; padding:12px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:4px; font-size:19px; font-variant-numeric:tabular-nums; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ color:var(--muted); background:#f8fafb; }}
    .chart-box {{ border:1px solid var(--line); border-radius:8px; background:#fbfcfd; padding:12px; }}
    svg {{ display:block; width:100%; height:auto; background:white; border-radius:6px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; color:var(--muted); font-size:13px; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
    .legend i {{ display:inline-block; width:18px; height:3px; border-radius:999px; }}
    .note {{ border-left:4px solid var(--amber); background:#fff9ef; color:#473923; padding:12px 14px; border-radius:6px; }}
    .files {{ display:grid; gap:8px; color:var(--muted); overflow-wrap:anywhere; font-size:14px; }}
    @media(max-width:820px) {{ .hero,.metric-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:28px; }} table {{ display:block; overflow-x:auto; white-space:nowrap; }} }}
  </style>
</head>
<body>
  <header class="top">
    <div class="wrap hero">
      <div>
        <div class="eyebrow">信用债 AI {html.escape(mode_title)}</div>
        <h1>四期限等权组合累计收益 {fmt_pct(portfolio['total_return'])}</h1>
        <p>用 3年、5年、10年、20年四个期限的三模型集成信号调仓。看多加仓、看空减仓、震荡不动；每个期限独立仓位，最后四个期限等权合成组合。{html.escape(mode_note)}</p>
      </div>
      <div class="box">
        <span>年化收益率</span>
        <strong>{fmt_pct(portfolio['annualized_return'])}</strong>
        <p>区间：{html.escape(portfolio['start_date'])} 至 {html.escape(portfolio['end_date'])}</p>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>核心结果</h2>
      <div class="metric-grid">
        <div class="metric"><span>累计收益率</span><strong>{fmt_pct(portfolio['total_return'])}</strong></div>
        <div class="metric"><span>年化收益率</span><strong>{fmt_pct(portfolio['annualized_return'])}</strong></div>
        <div class="metric"><span>最大回撤</span><strong>{fmt_pct(portfolio['max_drawdown'])}</strong></div>
        <div class="metric"><span>夏普比率</span><strong>{fmt_num(portfolio['sharpe'])}</strong></div>
        <div class="metric"><span>买满基准累计收益</span><strong>{fmt_pct(portfolio['buy_hold_total_return'])}</strong></div>
        <div class="metric"><span>50%静态仓位累计收益</span><strong>{fmt_pct(portfolio['neutral_total_return'])}</strong></div>
        <div class="metric"><span>平均仓位</span><strong>{fmt_pct(portfolio['average_position'], 1)}</strong></div>
        <div class="metric"><span>胜率</span><strong>{fmt_pct(portfolio['positive_day_ratio'], 1)}</strong></div>
        <div class="metric"><span>买满基准最大回撤</span><strong>{fmt_pct(portfolio['buy_hold_max_drawdown'])}</strong></div>
        <div class="metric"><span>50%静态最大回撤</span><strong>{fmt_pct(portfolio['neutral_max_drawdown'])}</strong></div>
        <div class="metric"><span>相对买满超额</span><strong>{fmt_pct(portfolio['excess_vs_buy_hold'])}</strong></div>
        <div class="metric"><span>相对50%静态超额</span><strong>{fmt_pct(portfolio['excess_vs_neutral'])}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>净值走势</h2>
      {svg_line_chart(report['portfolio_detail'])}
    </section>
    <section class="section">
      <h2>分期限结果</h2>
      <table>
        <thead><tr><th>期限</th><th>区间</th><th>累计收益</th><th>年化收益</th><th>最大回撤</th><th>买满基准</th><th>最终仓位</th><th>换手</th></tr></thead>
        <tbody>{tenor_rows}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>规则说明</h2>
      <div class="note">初始仓位 {fmt_pct(config['initial_position'], 1)}；每次加减 {fmt_pct(config['step'], 1)}；交易成本 {config['transaction_cost_bp']}bp；方向概率阈值 {fmt_pct(config['prob_threshold'], 0)}；方向概率相对震荡概率至少高 {fmt_pct(config['margin_vs_range'], 0)}。{html.escape(mode_note)}</div>
    </section>
    <section class="section">
      <h2>文件</h2>
      <div class="files">
        <div>组合明细：{html.escape(report['portfolio_csv'])}</div>
        <div>期限明细目录：{html.escape(report['tenor_detail_dir'])}</div>
        <div>汇总 CSV：{html.escape(report['summary_csv'])}</div>
        <div>JSON：{html.escape(report['json_path'])}</div>
      </div>
    </section>
  </main>
</body>
</html>"""
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_dirs = args.model_dir or DEFAULT_MODEL_DIRS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = out_dir / "tenor_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features, encoding="utf-8-sig")
    duration_map = parse_duration_map(args.duration_map)
    if args.rolling_summary:
        ensembles = load_rolling_ensemble_predictions(args.rolling_summary)
        mode = "rolling"
    else:
        ensembles = load_ensemble_predictions(model_dirs, features)
        mode = "full_history"

    tenor_metrics = []
    tenor_details = []
    for target_col, entity in sorted(ensembles.items(), key=lambda x: float(infer_tenor(x[0]) or 999)):
        metrics, detail = simulate_tenor(entity, features, duration_map, args)
        safe_name = entity["display_name"].replace(" ", "_").replace("+", "p")
        detail_path = detail_dir / f"{safe_name}_detail.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        metrics["detail_csv"] = str(detail_path)
        tenor_metrics.append(metrics)
        tenor_details.append(detail)

    portfolio_metrics, portfolio_detail = portfolio_from_tenors(tenor_details, args.annual_days)
    portfolio_csv = out_dir / "portfolio_detail.csv"
    portfolio_detail.to_csv(portfolio_csv, index=False, encoding="utf-8-sig")
    summary_csv = out_dir / "portfolio_summary.csv"
    pd.DataFrame(tenor_metrics).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    json_path = out_dir / "portfolio_backtest_report.json"
    html_path = out_dir / "portfolio_backtest_report.html"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_path": str(args.features),
        "config": {
            "mode": mode,
            "rolling_summary": args.rolling_summary,
            "initial_position": args.initial_position,
            "step": args.step,
            "transaction_cost_bp": args.transaction_cost_bp,
            "prob_threshold": args.prob_threshold,
            "margin_vs_range": args.margin_vs_range,
            "annual_days": args.annual_days,
            "include_carry": not args.no_carry,
            "duration_map": duration_map,
            "model_dirs": model_dirs,
        },
        "portfolio_metrics": portfolio_metrics,
        "tenor_metrics": tenor_metrics,
        "portfolio_detail": portfolio_detail.to_dict(orient="records"),
        "portfolio_csv": str(portfolio_csv),
        "tenor_detail_dir": str(detail_dir),
        "summary_csv": str(summary_csv),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, html_path)
    print(json.dumps({
        "HTML报告": str(html_path),
        "JSON报告": str(json_path),
        "组合明细": str(portfolio_csv),
        "汇总CSV": str(summary_csv),
        "累计收益率": portfolio_metrics["total_return"],
        "年化收益率": portfolio_metrics["annualized_return"],
        "最大回撤": portfolio_metrics["max_drawdown"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
