from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
SCRIPT_DIR = Path(__file__).absolute().parent
for item in [ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from creditbond_ai.position_backtest import _annualized_return, _max_drawdown, _sharpe
from run_rolling_portfolio_sweep import combo_namespace, parse_float_list
from run_simple_portfolio_backtest import (
    fmt_num,
    fmt_pct,
    load_rolling_ensemble_predictions,
    parse_duration_map,
    portfolio_from_tenors,
    simulate_tenor,
    svg_line_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select portfolio rules on early rolling folds and validate on later folds.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--rolling-summary", default="models/curve_2020_rolling_validation/rolling_validation_summary.csv")
    parser.add_argument("--out-dir", default="data/backtests/curve_2020_rolling_rule_locked")
    parser.add_argument("--selection-end", default="2025-01-03")
    parser.add_argument("--prob-thresholds", default="0.0,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--margins", default="0.0,0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--steps", default="0.02,0.05,0.1,0.15,0.2")
    parser.add_argument("--initial-positions", default="0.2,0.3,0.5,0.7")
    parser.add_argument("--transaction-cost-bp", type=float, default=0.0)
    parser.add_argument("--annual-days", type=int, default=252)
    parser.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    parser.add_argument("--no-carry", action="store_true")
    return parser.parse_args()


def subset_metrics(detail: pd.DataFrame, start_date: str | None, end_date: str | None, annual_days: int) -> tuple[dict[str, Any], pd.DataFrame]:
    work = detail.copy()
    work["date_ts"] = pd.to_datetime(work["date"])
    if start_date:
        work = work[work["date_ts"] >= pd.Timestamp(start_date)]
    if end_date:
        work = work[work["date_ts"] <= pd.Timestamp(end_date)]
    work = work.drop(columns=["date_ts"]).reset_index(drop=True)
    if work.empty:
        empty = {
            "start_date": "",
            "end_date": "",
            "periods": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "positive_day_ratio": 0.0,
            "buy_hold_total_return": 0.0,
            "neutral_total_return": 0.0,
            "average_position": 0.0,
        }
        return empty, work
    work["strategy_value"] = (1.0 + work["strategy_return"]).cumprod()
    work["buy_hold_value"] = (1.0 + work["buy_hold_return"]).cumprod()
    work["neutral_value"] = (1.0 + work["neutral_return"]).cumprod()
    periods = len(work)
    total = float(work["strategy_value"].iloc[-1] - 1.0)
    buy_hold_total = float(work["buy_hold_value"].iloc[-1] - 1.0)
    neutral_total = float(work["neutral_value"].iloc[-1] - 1.0)
    metrics = {
        "start_date": str(work["date"].iloc[0]),
        "end_date": str(work["next_date"].iloc[-1]),
        "periods": periods,
        "total_return": total,
        "annualized_return": _annualized_return(total, periods, annual_days),
        "max_drawdown": _max_drawdown(work["strategy_value"]),
        "sharpe": _sharpe(work["strategy_return"], annual_days),
        "positive_day_ratio": float((work["strategy_return"] > 0).mean()),
        "buy_hold_total_return": buy_hold_total,
        "neutral_total_return": neutral_total,
        "average_position": float(work["average_position"].mean()),
    }
    return metrics, work


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def table_rows(rows: list[dict[str, Any]], limit: int = 15) -> str:
    html_rows = []
    for idx, row in enumerate(rows[:limit], start=1):
        html_rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{fmt_pct(row['select_total_return'])}</td>
              <td>{fmt_pct(row['select_annualized_return'])}</td>
              <td>{fmt_pct(row['select_max_drawdown'])}</td>
              <td>{fmt_pct(row['eval_total_return'])}</td>
              <td>{fmt_pct(row['eval_annualized_return'])}</td>
              <td>{fmt_pct(row['eval_max_drawdown'])}</td>
              <td>{fmt_pct(row['eval_buy_hold_total_return'])}</td>
              <td>{fmt_pct(row['initial_position'], 1)}</td>
              <td>{fmt_pct(row['step'], 1)}</td>
              <td>{fmt_pct(row['prob_threshold'], 0)}</td>
              <td>{fmt_pct(row['margin_vs_range'], 0)}</td>
            </tr>
            """
        )
    return "\n".join(html_rows)


def write_html(report: dict[str, Any], path: Path) -> None:
    best = report["best"]
    base = report["baseline"]
    detail = report["best_eval_detail"]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>滚动规则锁定检验</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64717f; --line:#dce4ea; --paper:#fff; --back:#f4f6f8; --blue:#2f6f9f; --amber:#b7812f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    .wrap {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; }}
    .top {{ background:var(--paper); border-bottom:1px solid var(--line); }}
    .hero {{ padding:34px 0 26px; display:grid; grid-template-columns:1.2fr .9fr; gap:24px; align-items:end; }}
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
    th:first-child,td:first-child {{ text-align:left; }}
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
        <div class="eyebrow">信用债 AI 规则锁定检验</div>
        <h1>锁定规则后段累计收益 {fmt_pct(best['eval_total_return'])}</h1>
        <p>先用 {html.escape(report['selection_range'])} 选择仓位规则，再把规则锁死到 {html.escape(report['evaluation_range'])} 检验。</p>
      </div>
      <div class="box">
        <p>原规则后段累计收益 {fmt_pct(base['eval_total_return'])}；锁定后最佳规则为：初始 {fmt_pct(best['initial_position'], 1)}，步长 {fmt_pct(best['step'], 1)}，概率阈值 {fmt_pct(best['prob_threshold'], 0)}，边际 {fmt_pct(best['margin_vs_range'], 0)}。</p>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>锁定检验结果</h2>
      <div class="metric-grid">
        <div class="metric"><span>选择段累计收益</span><strong>{fmt_pct(best['select_total_return'])}</strong></div>
        <div class="metric"><span>选择段年化</span><strong>{fmt_pct(best['select_annualized_return'])}</strong></div>
        <div class="metric"><span>检验段累计收益</span><strong>{fmt_pct(best['eval_total_return'])}</strong></div>
        <div class="metric"><span>检验段年化</span><strong>{fmt_pct(best['eval_annualized_return'])}</strong></div>
        <div class="metric"><span>检验段最大回撤</span><strong>{fmt_pct(best['eval_max_drawdown'])}</strong></div>
        <div class="metric"><span>检验段买满基准</span><strong>{fmt_pct(best['eval_buy_hold_total_return'])}</strong></div>
        <div class="metric"><span>原规则检验段收益</span><strong>{fmt_pct(base['eval_total_return'])}</strong></div>
        <div class="metric"><span>检验段夏普</span><strong>{fmt_num(best['eval_sharpe'])}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>检验段净值</h2>
      {svg_line_chart(detail)}
    </section>
    <section class="section">
      <h2>按选择段排名的前 15 条规则</h2>
      <table>
        <thead><tr><th>#</th><th>选择累计</th><th>选择年化</th><th>选择回撤</th><th>检验累计</th><th>检验年化</th><th>检验回撤</th><th>检验买满</th><th>初始</th><th>步长</th><th>阈值</th><th>边际</th></tr></thead>
        <tbody>{table_rows(report['top_rows'])}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>怎么读</h2>
      <div class="note">如果检验段收益明显高于原规则，说明规则优化有实际价值；如果仍然输给买满基准，说明仓位规则只能改善控仓效率，真正要追上买满还要改模型目标、标签和数据质量。</div>
    </section>
    <section class="section">
      <h2>文件</h2>
      <div class="files">
        <div>全量规则 CSV：{html.escape(report['locked_csv'])}</div>
        <div>最佳检验段明细：{html.escape(report['best_eval_detail_csv'])}</div>
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
    ensembles = load_rolling_ensemble_predictions(args.rolling_summary)

    rows = []
    eval_details: dict[int, pd.DataFrame] = {}
    combo_id = 0
    for initial in parse_float_list(args.initial_positions):
        for step in parse_float_list(args.steps):
            for threshold in parse_float_list(args.prob_thresholds):
                for margin in parse_float_list(args.margins):
                    combo_id += 1
                    combo_args = argparse.Namespace(
                        initial_position=initial,
                        step=step,
                        transaction_cost_bp=args.transaction_cost_bp,
                        prob_threshold=threshold,
                        margin_vs_range=margin,
                        annual_days=args.annual_days,
                        no_carry=args.no_carry,
                    )
                    tenor_details = []
                    for _, entity in sorted(ensembles.items()):
                        _, detail = simulate_tenor(entity, features, duration_map, combo_args)
                        tenor_details.append(detail)
                    _, portfolio_detail = portfolio_from_tenors(tenor_details, args.annual_days)
                    select_metrics, _ = subset_metrics(portfolio_detail, None, args.selection_end, args.annual_days)
                    eval_start = (pd.Timestamp(args.selection_end) + pd.Timedelta(days=1)).date().isoformat()
                    eval_metrics, eval_detail = subset_metrics(portfolio_detail, eval_start, None, args.annual_days)
                    row = {
                        "combo_id": combo_id,
                        "initial_position": initial,
                        "step": step,
                        "prob_threshold": threshold,
                        "margin_vs_range": margin,
                        **prefixed("select", select_metrics),
                        **prefixed("eval", eval_metrics),
                    }
                    rows.append(row)
                    eval_details[combo_id] = eval_detail

    table = pd.DataFrame(rows)
    table = table.sort_values(["select_total_return", "select_annualized_return", "select_max_drawdown"], ascending=[False, False, False])
    locked_csv = out_dir / "rolling_rule_locked_validation.csv"
    table.to_csv(locked_csv, index=False, encoding="utf-8-sig")
    best = table.iloc[0].to_dict()
    baseline = table[
        table["initial_position"].round(6).eq(0.5)
        & table["step"].round(6).eq(0.05)
        & table["prob_threshold"].round(6).eq(0.45)
        & table["margin_vs_range"].round(6).eq(0.03)
    ].iloc[0].to_dict()
    best_eval_detail = eval_details[int(best["combo_id"])]
    best_eval_detail_csv = out_dir / "best_locked_eval_detail.csv"
    best_eval_detail.to_csv(best_eval_detail_csv, index=False, encoding="utf-8-sig")

    json_path = out_dir / "rolling_rule_locked_validation.json"
    html_path = out_dir / "rolling_rule_locked_validation_report.html"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_end": args.selection_end,
        "selection_range": f"{table.iloc[0]['select_start_date']} 至 {table.iloc[0]['select_end_date']}",
        "evaluation_range": f"{best['eval_start_date']} 至 {best['eval_end_date']}",
        "config": vars(args),
        "experiment_count": len(table),
        "best": best,
        "baseline": baseline,
        "top_rows": table.head(15).to_dict(orient="records"),
        "best_eval_detail": best_eval_detail.to_dict(orient="records"),
        "locked_csv": str(locked_csv),
        "best_eval_detail_csv": str(best_eval_detail_csv),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, html_path)
    print(json.dumps({
        "HTML报告": str(html_path),
        "规则CSV": str(locked_csv),
        "实验数量": len(table),
        "选择段截止": args.selection_end,
        "最佳规则": {
            "initial_position": best["initial_position"],
            "step": best["step"],
            "prob_threshold": best["prob_threshold"],
            "margin_vs_range": best["margin_vs_range"],
        },
        "最佳规则检验段累计收益率": best["eval_total_return"],
        "原规则检验段累计收益率": baseline["eval_total_return"],
        "最佳规则检验段买满基准": best["eval_buy_hold_total_return"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
