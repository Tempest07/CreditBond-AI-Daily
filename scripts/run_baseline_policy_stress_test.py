from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
SCRIPT_DIR = Path(__file__).absolute().parent
for item in [ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from creditbond_ai.position_backtest import _annualized_return, _max_drawdown, _prepare_asset_returns, _sharpe
from run_simple_portfolio_backtest import (
    apply_rule,
    fmt_num,
    fmt_pct,
    load_rolling_ensemble_predictions,
    parse_duration_map,
    svg_line_chart,
)


CREDIT_COLS = {
    "3": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):3\u5e74",
    "5": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):5\u5e74",
    "10": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):10\u5e74",
    "20": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA+):20\u5e74",
}

TREASURY_COLS = {
    "3": "\u4e2d\u503a\u56fd\u503a\u5230\u671f\u6536\u76ca\u7387:3\u5e74",
    "5": "\u4e2d\u503a\u56fd\u503a\u5230\u671f\u6536\u76ca\u7387:5\u5e74",
    "10": "\u4e2d\u503a\u56fd\u503a\u5230\u671f\u6536\u76ca\u7387:10\u5e74",
    "20": "\u4e2d\u503a\u56fd\u503a\u5230\u671f\u6536\u76ca\u7387:20\u5e74",
}


@dataclass
class Policy:
    name: str
    kind: str
    base: float
    step: float = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test baseline-position policies on rolling predictions.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--rolling-summary", default="models/curve_2020_rolling_validation/rolling_validation_summary.csv")
    parser.add_argument("--out-dir", default="data/backtests/curve_2020_baseline_policy_stress")
    parser.add_argument("--prob-threshold", type=float, default=0.45)
    parser.add_argument("--margin-vs-range", type=float, default=0.03)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--annual-days", type=int, default=252)
    parser.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    parser.add_argument("--no-carry", action="store_true")
    return parser.parse_args()


def infer_tenor(target_col: str) -> str:
    for tenor in ["20", "10", "5", "3"]:
        if f":{tenor}年" in target_col:
            return tenor
    return ""


def rolling_percentile_last(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    def pct(values: np.ndarray) -> float:
        s = pd.Series(values)
        return float(s.rank(pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=min_periods).apply(pct, raw=True)


def build_regime_frame(features: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"])
    credit_cols = [col for col in CREDIT_COLS.values() if col in work.columns]
    spread_cols = []
    for tenor, credit_col in CREDIT_COLS.items():
        treasury_col = TREASURY_COLS.get(tenor)
        if credit_col in work.columns and treasury_col in work.columns:
            spread_name = f"spread_{tenor}"
            work[spread_name] = (work[credit_col] - work[treasury_col]) * 100.0
            spread_cols.append(spread_name)
    if not credit_cols:
        raise ValueError("特征表中找不到信用债收益率列。")
    work["avg_credit_yield"] = work[credit_cols].mean(axis=1)
    work["avg_credit_spread_bp"] = work[spread_cols].mean(axis=1) if spread_cols else 0.0
    work["yield_change_20d_bp"] = (work["avg_credit_yield"] - work["avg_credit_yield"].shift(20)) * 100.0
    work["spread_change_20d_bp"] = work["avg_credit_spread_bp"] - work["avg_credit_spread_bp"].shift(20)
    work["yield_pct_1y"] = rolling_percentile_last(work["avg_credit_yield"])
    work["spread_pct_1y"] = rolling_percentile_last(work["avg_credit_spread_bp"])

    conditions = [
        (work["yield_change_20d_bp"] <= -5.0) & (work["spread_change_20d_bp"] <= 2.0),
        (work["yield_change_20d_bp"] >= 5.0) | (work["spread_change_20d_bp"] >= 5.0),
    ]
    work["regime"] = np.select(conditions, ["顺风", "逆风"], default="震荡")

    dynamic = pd.Series(0.6, index=work.index, dtype=float)
    dynamic[(work["yield_change_20d_bp"] >= 10.0) | (work["spread_change_20d_bp"] >= 8.0)] = 0.3
    dynamic[(work["spread_pct_1y"] >= 0.65) & (work["yield_change_20d_bp"] <= 5.0)] = 0.8
    dynamic[(work["yield_pct_1y"] <= 0.30) & (work["spread_pct_1y"] <= 0.40)] = 0.4
    work["dynamic_base"] = dynamic.fillna(0.5).clip(0.2, 0.9)
    return work[
        [
            "date",
            "avg_credit_yield",
            "avg_credit_spread_bp",
            "yield_change_20d_bp",
            "spread_change_20d_bp",
            "regime",
            "dynamic_base",
        ]
    ].copy()


def load_tenor_rows(
    features: pd.DataFrame,
    rolling_summary: str | Path,
    duration_map: dict[str, float],
    prob_threshold: float,
    margin_vs_range: float,
    annual_days: int,
    include_carry: bool,
) -> pd.DataFrame:
    ensembles = load_rolling_ensemble_predictions(rolling_summary)
    rows = []
    for target_col, entity in sorted(ensembles.items()):
        pred = apply_rule(entity["pred"], prob_threshold, margin_vs_range)
        tenor = infer_tenor(target_col)
        duration = duration_map.get(tenor, max(1.0, float(tenor or 4) * 0.75))
        asset = _prepare_asset_returns(
            features=features,
            target_col=target_col,
            duration=duration,
            annual_days=annual_days,
            include_carry=include_carry,
        )
        merged = pred.merge(asset, on="date", how="inner").sort_values("date")
        merged["tenor"] = tenor
        merged["target_col"] = target_col
        rows.append(
            merged[
                [
                    "date",
                    "next_date",
                    "tenor",
                    "target_col",
                    "signal",
                    "prediction",
                    "bond_return_proxy",
                ]
            ].copy()
        )
    return pd.concat(rows, ignore_index=True)


def policy_list(step: float) -> list[Policy]:
    return [
        Policy("静态20%", "static", 0.2, step),
        Policy("静态50%", "static", 0.5, step),
        Policy("静态80%", "static", 0.8, step),
        Policy("静态100%", "static", 1.0, step),
        Policy("AI对称20%", "symmetric", 0.2, step),
        Policy("AI对称50%", "symmetric", 0.5, step),
        Policy("AI对称80%", "symmetric", 0.8, step),
        Policy("AI风险开关80%", "risk_switch", 0.8, step),
        Policy("AI风险开关100%", "risk_switch", 1.0, step),
        Policy("AI机会捕捉20%", "opportunity", 0.2, step),
        Policy("动态底仓风险开关", "dynamic_risk_switch", 0.6, step),
    ]


def base_for_date(policy: Policy, row: pd.Series) -> float:
    if policy.kind == "dynamic_risk_switch":
        return float(row["dynamic_base"])
    return float(policy.base)


def update_position(policy: Policy, previous: float, signal: int, base: float) -> float:
    if policy.kind == "static":
        return base
    if policy.kind == "symmetric":
        if signal > 0:
            return min(1.0, previous + policy.step)
        if signal < 0:
            return max(0.0, previous - policy.step)
        return previous
    if policy.kind in {"risk_switch", "dynamic_risk_switch"}:
        if signal < 0:
            return max(0.0, previous - policy.step)
        if previous < base:
            return min(base, previous + policy.step)
        if previous > base:
            return max(base, previous - policy.step)
        return base
    if policy.kind == "opportunity":
        if signal > 0:
            return min(1.0, previous + policy.step)
        if signal < 0:
            return max(base, previous - policy.step)
        if previous > base:
            return max(base, previous - policy.step)
        return base
    return previous


def simulate_policy(policy: Policy, tenor_rows: pd.DataFrame, regime: pd.DataFrame, annual_days: int) -> tuple[dict[str, Any], pd.DataFrame]:
    work = tenor_rows.merge(regime, on="date", how="left").sort_values(["date", "tenor"]).copy()
    positions: dict[str, float] = {}
    records = []
    for date, day in work.groupby("date", sort=True):
        returns = []
        position_values = []
        trade_sum = 0.0
        day_regime = str(day["regime"].iloc[0])
        next_date = pd.Timestamp(day["next_date"].iloc[0]).date().isoformat()
        dynamic_base = float(day["dynamic_base"].iloc[0])
        for _, row in day.iterrows():
            tenor = str(row["tenor"])
            base = base_for_date(policy, row)
            previous = positions.get(tenor, base)
            position = update_position(policy, previous, int(row["signal"]), base)
            positions[tenor] = position
            trade_sum += abs(position - previous)
            returns.append(position * float(row["bond_return_proxy"]))
            position_values.append(position)
        records.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "next_date": next_date,
                "regime": day_regime,
                "dynamic_base": dynamic_base,
                "strategy_return": float(np.mean(returns)),
                "average_position": float(np.mean(position_values)),
                "daily_turnover": float(trade_sum),
            }
        )
    detail = pd.DataFrame(records)
    detail["strategy_value"] = (1.0 + detail["strategy_return"]).cumprod()
    metrics = metrics_from_detail(detail)
    metrics.update({"policy": policy.name, "kind": policy.kind, "base": policy.base, "step": policy.step})
    return metrics, detail


def metrics_from_detail(detail: pd.DataFrame) -> dict[str, Any]:
    if detail.empty:
        return {
            "start_date": "",
            "end_date": "",
            "periods": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "positive_day_ratio": 0.0,
            "average_position": 0.0,
            "turnover": 0.0,
        }
    periods = len(detail)
    value = (1.0 + detail["strategy_return"]).cumprod()
    total = float(value.iloc[-1] - 1.0)
    return {
        "start_date": str(detail["date"].iloc[0]),
        "end_date": str(detail["next_date"].iloc[-1]),
        "periods": periods,
        "total_return": total,
        "annualized_return": _annualized_return(total, periods, 252),
        "max_drawdown": _max_drawdown(value),
        "sharpe": _sharpe(detail["strategy_return"], 252),
        "positive_day_ratio": float((detail["strategy_return"] > 0).mean()),
        "average_position": float(detail["average_position"].mean()),
        "turnover": float(detail["daily_turnover"].sum()),
    }


def regime_rows(policy_details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for policy, detail in policy_details.items():
        for regime_name, sub in detail.groupby("regime"):
            metrics = metrics_from_detail(sub.reset_index(drop=True))
            rows.append({"policy": policy, "regime": regime_name, **metrics})
    return pd.DataFrame(rows)


def table_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(
            f"""
            <tr>
              <td>{html.escape(row['policy'])}</td>
              <td>{fmt_pct(row['total_return'])}</td>
              <td>{fmt_pct(row['annualized_return'])}</td>
              <td>{fmt_pct(row['max_drawdown'])}</td>
              <td>{fmt_pct(row['average_position'], 1)}</td>
              <td>{fmt_pct(row['positive_day_ratio'], 1)}</td>
              <td>{fmt_num(row['sharpe'])}</td>
              <td>{fmt_num(row['turnover'], 2)}</td>
            </tr>
            """
        )
    return "\n".join(out)


def regime_table_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(
            f"""
            <tr>
              <td>{html.escape(row['policy'])}</td>
              <td>{html.escape(row['regime'])}</td>
              <td>{fmt_pct(row['total_return'])}</td>
              <td>{fmt_pct(row['annualized_return'])}</td>
              <td>{fmt_pct(row['max_drawdown'])}</td>
              <td>{fmt_pct(row['average_position'], 1)}</td>
            </tr>
            """
        )
    return "\n".join(out)


def write_html(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    best = summary[0]
    static100 = next((row for row in summary if row["policy"] == "静态100%"), None)
    static50 = next((row for row in summary if row["policy"] == "静态50%"), None)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>基准仓位压力测试</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64717f; --line:#dce4ea; --paper:#fff; --back:#f4f6f8; --blue:#2f6f9f; --amber:#b7812f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    .wrap {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; }}
    .top {{ background:var(--paper); border-bottom:1px solid var(--line); }}
    .hero {{ padding:34px 0 26px; display:grid; grid-template-columns:1.25fr .85fr; gap:24px; align-items:end; }}
    .eyebrow {{ color:var(--blue); font-weight:800; font-size:14px; margin-bottom:8px; }}
    h1 {{ margin:0; font-size:34px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    p {{ margin:10px 0 0; color:var(--muted); }}
    .box,.section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:18px; }}
    .main {{ display:grid; gap:18px; padding:22px 0 46px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .metric {{ background:#f8fafb; border-top:3px solid var(--blue); border-radius:6px; padding:12px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:4px; font-size:19px; font-variant-numeric:tabular-nums; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:9px 7px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ color:var(--muted); background:#f8fafb; }}
    .note {{ border-left:4px solid var(--amber); background:#fff9ef; color:#473923; padding:12px 14px; border-radius:6px; }}
    .chart-box {{ border:1px solid var(--line); border-radius:8px; background:#fbfcfd; padding:12px; }}
    svg {{ display:block; width:100%; height:auto; background:white; border-radius:6px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; color:var(--muted); font-size:13px; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
    .legend i {{ display:inline-block; width:18px; height:3px; border-radius:999px; }}
    .files {{ display:grid; gap:8px; color:var(--muted); overflow-wrap:anywhere; font-size:14px; }}
    @media(max-width:860px) {{ .hero,.metric-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:28px; }} table {{ display:block; overflow-x:auto; white-space:nowrap; }} }}
  </style>
</head>
<body>
  <header class="top">
    <div class="wrap hero">
      <div>
        <div class="eyebrow">信用债 AI 基准仓位压力测试</div>
        <h1>不预设牛市，比较不同底仓世界观</h1>
        <p>同一批滚动样本外预测，同时测试静态底仓、对称加减仓、风险开关、机会捕捉和动态底仓，并拆分顺风/逆风/震荡市场状态。</p>
      </div>
      <div class="box">
        <p>本轮最佳：{html.escape(best['policy'])}，累计收益 {fmt_pct(best['total_return'])}，最大回撤 {fmt_pct(best['max_drawdown'])}。</p>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>关键对比</h2>
      <div class="metric-grid">
        <div class="metric"><span>最佳策略</span><strong>{html.escape(best['policy'])}</strong></div>
        <div class="metric"><span>最佳累计收益</span><strong>{fmt_pct(best['total_return'])}</strong></div>
        <div class="metric"><span>最佳最大回撤</span><strong>{fmt_pct(best['max_drawdown'])}</strong></div>
        <div class="metric"><span>最佳平均仓位</span><strong>{fmt_pct(best['average_position'], 1)}</strong></div>
        <div class="metric"><span>静态100%收益</span><strong>{fmt_pct(static100['total_return']) if static100 else '无'}</strong></div>
        <div class="metric"><span>静态100%回撤</span><strong>{fmt_pct(static100['max_drawdown']) if static100 else '无'}</strong></div>
        <div class="metric"><span>静态50%收益</span><strong>{fmt_pct(static50['total_return']) if static50 else '无'}</strong></div>
        <div class="metric"><span>静态50%回撤</span><strong>{fmt_pct(static50['max_drawdown']) if static50 else '无'}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>所有策略总览</h2>
      <table>
        <thead><tr><th>策略</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>平均仓位</th><th>胜率</th><th>夏普</th><th>换手</th></tr></thead>
        <tbody>{table_rows(summary)}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>按市场状态拆分</h2>
      <table>
        <thead><tr><th>策略</th><th>市场状态</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>平均仓位</th></tr></thead>
        <tbody>{regime_table_rows(report['regime_summary'])}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>最佳策略净值</h2>
      {svg_line_chart(report['best_detail'])}
    </section>
    <section class="section">
      <h2>怎么读</h2>
      <div class="note">这份报告的目的不是选择历史最好看的仓位，而是检查不同底仓假设在顺风、逆风、震荡中是否脆弱。如果静态100%仍然长期领先，说明当前样本仍偏牛；如果动态/风险开关在逆风段明显降低回撤，才说明 AI 真有风险管理价值。</div>
    </section>
    <section class="section">
      <h2>文件</h2>
      <div class="files">
        <div>汇总 CSV：{html.escape(report['summary_csv'])}</div>
        <div>市场状态 CSV：{html.escape(report['regime_csv'])}</div>
        <div>最佳策略明细：{html.escape(report['best_detail_csv'])}</div>
        <div>JSON：{html.escape(report['json_path'])}</div>
      </div>
    </section>
  </main>
</body>
</html>"""
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features, encoding="utf-8-sig")
    duration_map = parse_duration_map(args.duration_map)
    regime = build_regime_frame(features)
    tenor_rows = load_tenor_rows(
        features=features,
        rolling_summary=args.rolling_summary,
        duration_map=duration_map,
        prob_threshold=args.prob_threshold,
        margin_vs_range=args.margin_vs_range,
        annual_days=args.annual_days,
        include_carry=not args.no_carry,
    )
    policies = policy_list(args.step)
    summary_rows = []
    details: dict[str, pd.DataFrame] = {}
    for policy in policies:
        metrics, detail = simulate_policy(policy, tenor_rows, regime, args.annual_days)
        summary_rows.append(metrics)
        details[policy.name] = detail

    summary = pd.DataFrame(summary_rows).sort_values(["total_return", "max_drawdown"], ascending=[False, False])
    regime_summary = regime_rows(details).sort_values(["policy", "regime"])
    summary_csv = out_dir / "baseline_policy_summary.csv"
    regime_csv = out_dir / "baseline_policy_regime_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    regime_summary.to_csv(regime_csv, index=False, encoding="utf-8-sig")

    best_policy = str(summary.iloc[0]["policy"])
    best_detail = details[best_policy]
    best_detail_csv = out_dir / "best_policy_detail.csv"
    best_detail.to_csv(best_detail_csv, index=False, encoding="utf-8-sig")
    chart_detail = best_detail[["date", "next_date", "strategy_value"]].copy()
    if "静态100%" in details:
        static_full = details["静态100%"][["date", "strategy_value"]].rename(columns={"strategy_value": "buy_hold_value"})
        chart_detail = chart_detail.merge(static_full, on="date", how="left")
    else:
        chart_detail["buy_hold_value"] = chart_detail["strategy_value"]
    if "静态50%" in details:
        static_half = details["静态50%"][["date", "strategy_value"]].rename(columns={"strategy_value": "neutral_value"})
        chart_detail = chart_detail.merge(static_half, on="date", how="left")
    else:
        chart_detail["neutral_value"] = chart_detail["strategy_value"]

    json_path = out_dir / "baseline_policy_stress_report.json"
    html_path = out_dir / "baseline_policy_stress_report.html"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "summary": summary.to_dict(orient="records"),
        "regime_summary": regime_summary.to_dict(orient="records"),
        "best_detail": chart_detail.to_dict(orient="records"),
        "summary_csv": str(summary_csv),
        "regime_csv": str(regime_csv),
        "best_detail_csv": str(best_detail_csv),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, html_path)
    print(json.dumps({
        "HTML报告": str(html_path),
        "汇总CSV": str(summary_csv),
        "市场状态CSV": str(regime_csv),
        "最佳策略": best_policy,
        "最佳累计收益率": float(summary.iloc[0]["total_return"]),
        "最佳最大回撤": float(summary.iloc[0]["max_drawdown"]),
        "策略数量": len(summary),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
