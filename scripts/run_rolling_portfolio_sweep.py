from __future__ import annotations

import argparse
import html
import json
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

from run_simple_portfolio_backtest import (
    fmt_num,
    fmt_pct,
    load_rolling_ensemble_predictions,
    parse_duration_map,
    portfolio_from_tenors,
    simulate_tenor,
    svg_line_chart,
)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep portfolio rules on rolling out-of-sample predictions.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--rolling-summary", default="models/curve_2020_rolling_validation/rolling_validation_summary.csv")
    parser.add_argument("--out-dir", default="data/backtests/curve_2020_rolling_portfolio_sweep")
    parser.add_argument("--prob-thresholds", default="0.0,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--margins", default="0.0,0.02,0.03,0.05,0.08")
    parser.add_argument("--steps", default="0.02,0.05,0.1,0.15")
    parser.add_argument("--initial-positions", default="0.3,0.5,0.7")
    parser.add_argument("--transaction-cost-bp", type=float, default=0.0)
    parser.add_argument("--annual-days", type=int, default=252)
    parser.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    parser.add_argument("--no-carry", action="store_true")
    return parser.parse_args()


def combo_namespace(base: argparse.Namespace, initial: float, step: float, threshold: float, margin: float) -> argparse.Namespace:
    return argparse.Namespace(
        initial_position=initial,
        step=step,
        transaction_cost_bp=base.transaction_cost_bp,
        prob_threshold=threshold,
        margin_vs_range=margin,
        annual_days=base.annual_days,
        no_carry=base.no_carry,
    )


def table_rows(rows: list[dict[str, Any]], limit: int = 20) -> str:
    html_rows = []
    for idx, row in enumerate(rows[:limit], start=1):
        html_rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{fmt_pct(row['total_return'])}</td>
              <td>{fmt_pct(row['annualized_return'])}</td>
              <td>{fmt_pct(row['max_drawdown'])}</td>
              <td>{fmt_pct(row['buy_hold_total_return'])}</td>
              <td>{fmt_pct(row['neutral_total_return'])}</td>
              <td>{fmt_pct(row['excess_vs_buy_hold'])}</td>
              <td>{fmt_pct(row['excess_vs_neutral'])}</td>
              <td>{fmt_pct(row['initial_position'], 1)}</td>
              <td>{fmt_pct(row['step'], 1)}</td>
              <td>{fmt_pct(row['prob_threshold'], 0)}</td>
              <td>{fmt_pct(row['margin_vs_range'], 0)}</td>
              <td>{fmt_pct(row['average_position'], 1)}</td>
            </tr>
            """
        )
    return "\n".join(html_rows)


def write_html(report: dict[str, Any], path: Path) -> None:
    best = report["best"]
    baseline = report.get("baseline")
    baseline_text = "未找到原始规则"
    if baseline:
        baseline_text = (
            f"原始规则累计收益 {fmt_pct(baseline['total_return'])}，"
            f"年化 {fmt_pct(baseline['annualized_return'])}，"
            f"最大回撤 {fmt_pct(baseline['max_drawdown'])}。"
        )
    best_detail = report["best_detail"]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>滚动样本外仓位规则扫描</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64717f; --line:#dce4ea; --paper:#fff; --back:#f4f6f8; --blue:#2f6f9f; --amber:#b7812f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    .wrap {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; }}
    .top {{ background:var(--paper); border-bottom:1px solid var(--line); }}
    .hero {{ padding:34px 0 26px; display:grid; grid-template-columns:1.3fr .8fr; gap:24px; align-items:end; }}
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
        <div class="eyebrow">信用债 AI 滚动样本外规则扫描</div>
        <h1>最佳规则累计收益 {fmt_pct(best['total_return'])}</h1>
        <p>固定滚动预测不变，只扫描“概率阈值、震荡边际、仓位步长、初始仓位”。这不是重新训练模型，而是在检验现有预测如何转成仓位更合理。</p>
      </div>
      <div class="box">
        <p>{html.escape(baseline_text)}</p>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>最佳规则</h2>
      <div class="metric-grid">
        <div class="metric"><span>累计收益率</span><strong>{fmt_pct(best['total_return'])}</strong></div>
        <div class="metric"><span>年化收益率</span><strong>{fmt_pct(best['annualized_return'])}</strong></div>
        <div class="metric"><span>最大回撤</span><strong>{fmt_pct(best['max_drawdown'])}</strong></div>
        <div class="metric"><span>夏普比率</span><strong>{fmt_num(best['sharpe'])}</strong></div>
        <div class="metric"><span>初始仓位</span><strong>{fmt_pct(best['initial_position'], 1)}</strong></div>
        <div class="metric"><span>每次加减仓</span><strong>{fmt_pct(best['step'], 1)}</strong></div>
        <div class="metric"><span>概率阈值</span><strong>{fmt_pct(best['prob_threshold'], 0)}</strong></div>
        <div class="metric"><span>震荡边际</span><strong>{fmt_pct(best['margin_vs_range'], 0)}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>最佳规则净值</h2>
      {svg_line_chart(best_detail)}
    </section>
    <section class="section">
      <h2>Top 20 规则</h2>
      <table>
        <thead>
          <tr><th>#</th><th>累计</th><th>年化</th><th>回撤</th><th>买满</th><th>50%静态</th><th>相对买满</th><th>相对静态</th><th>初始</th><th>步长</th><th>概率阈值</th><th>边际</th><th>平均仓位</th></tr>
        </thead>
        <tbody>{table_rows(report['top_rows'])}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>重要提醒</h2>
      <div class="note">这是在同一段滚动样本外预测上做的后验规则扫描，能说明“现有预测还有多少仓位规则改进空间”，但不能直接当作未来可复制收益。更严肃的下一步是：用前几折选择规则，只在最后几折做锁定规则验证。</div>
    </section>
    <section class="section">
      <h2>文件</h2>
      <div class="files">
        <div>规则扫描 CSV：{html.escape(report['sweep_csv'])}</div>
        <div>最佳组合明细：{html.escape(report['best_detail_csv'])}</div>
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
    detail_by_key: dict[int, pd.DataFrame] = {}
    combo_id = 0
    for initial in parse_float_list(args.initial_positions):
        for step in parse_float_list(args.steps):
            for threshold in parse_float_list(args.prob_thresholds):
                for margin in parse_float_list(args.margins):
                    combo_id += 1
                    combo_args = combo_namespace(args, initial, step, threshold, margin)
                    tenor_details = []
                    for _, entity in sorted(ensembles.items()):
                        _, detail = simulate_tenor(entity, features, duration_map, combo_args)
                        tenor_details.append(detail)
                    metrics, portfolio_detail = portfolio_from_tenors(tenor_details, args.annual_days)
                    row = {
                        "combo_id": combo_id,
                        "initial_position": initial,
                        "step": step,
                        "prob_threshold": threshold,
                        "margin_vs_range": margin,
                        **metrics,
                    }
                    rows.append(row)
                    detail_by_key[combo_id] = portfolio_detail

    sweep = pd.DataFrame(rows).sort_values(["total_return", "annualized_return", "max_drawdown"], ascending=[False, False, False])
    sweep_csv = out_dir / "rolling_portfolio_rule_sweep.csv"
    sweep.to_csv(sweep_csv, index=False, encoding="utf-8-sig")
    best = sweep.iloc[0].to_dict()
    best_detail = detail_by_key[int(best["combo_id"])]
    best_detail_csv = out_dir / "best_portfolio_detail.csv"
    best_detail.to_csv(best_detail_csv, index=False, encoding="utf-8-sig")

    baseline = sweep[
        sweep["initial_position"].round(6).eq(0.5)
        & sweep["step"].round(6).eq(0.05)
        & sweep["prob_threshold"].round(6).eq(0.45)
        & sweep["margin_vs_range"].round(6).eq(0.03)
    ]
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else None
    json_path = out_dir / "rolling_portfolio_rule_sweep.json"
    html_path = out_dir / "rolling_portfolio_rule_sweep_report.html"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "experiment_count": len(sweep),
        "best": best,
        "baseline": baseline_row,
        "top_rows": sweep.head(20).to_dict(orient="records"),
        "best_detail": best_detail.to_dict(orient="records"),
        "sweep_csv": str(sweep_csv),
        "best_detail_csv": str(best_detail_csv),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, html_path)
    print(json.dumps({
        "HTML报告": str(html_path),
        "扫描CSV": str(sweep_csv),
        "最佳明细": str(best_detail_csv),
        "实验数量": len(sweep),
        "最佳累计收益率": best["total_return"],
        "最佳年化收益率": best["annualized_return"],
        "最佳最大回撤": best["max_drawdown"],
        "原规则累计收益率": baseline_row["total_return"] if baseline_row else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
