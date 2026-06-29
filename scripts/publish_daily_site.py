from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from daily_report_utils import (
    fmt_bp,
    fmt_prob,
    fmt_yield,
    load_report,
    report_date,
    resolve_path,
    signal_tone,
    summary,
    tenor_rows,
)


DEFAULT_JSON = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.json"
DEFAULT_HTML = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Cloudflare Pages-ready site from daily report outputs.")
    parser.add_argument("--json-report", default=DEFAULT_JSON)
    parser.add_argument("--html-report", default=DEFAULT_HTML)
    parser.add_argument("--pdf", default="")
    parser.add_argument("--out-dir", default="output/site")
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def badge(signal: str) -> str:
    tone = signal_tone(signal)
    return f'<span class="badge {tone}">{html.escape(signal)}</span>'


def build_index(report: dict, has_pdf: bool) -> str:
    date_text = report_date(report)
    s = summary(report)
    row_html = []
    for row in tenor_rows(report):
        row_html.append(
            f"""
            <tr>
              <td>{html.escape(row['label'])}</td>
              <td>{html.escape(row['latest_date'])}</td>
              <td>{badge(row['signal'])}</td>
              <td>{html.escape(row['strict_signal'])}</td>
              <td>{fmt_yield(row['latest_yield'])}</td>
              <td>{fmt_bp(row['change_1d_bp'])}</td>
              <td>{fmt_bp(row['change_5d_bp'])}</td>
              <td>{fmt_bp(row['change_20d_bp'])}</td>
              <td>{fmt_prob(row['prob_bearish'])}</td>
              <td>{fmt_prob(row['prob_bullish'])}</td>
              <td>{fmt_prob(row['prob_range'])}</td>
            </tr>
            """
        )
    pdf_link = '<a class="button" href="latest/report.pdf">下载 PDF</a>' if has_pdf else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 日报</title>
  <style>
    :root {{
      --ink:#17202a; --muted:#667789; --line:#dde5ec; --back:#f5f7f8; --paper:#fff;
      --red:#c74343; --red-soft:#f8dddd; --green:#238a62; --green-soft:#dff2e9; --gold:#b7812f; --gold-soft:#f8edda;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; }}
    .wrap {{ width:min(1080px, calc(100% - 32px)); margin:0 auto; }}
    header {{ background:var(--paper); border-bottom:1px solid var(--line); padding:32px 0 24px; }}
    .eyebrow {{ color:#2f6f9f; font-weight:800; margin-bottom:8px; }}
    h1 {{ margin:0; font-size:34px; letter-spacing:0; }}
    .summary {{ margin-top:16px; display:grid; grid-template-columns:minmax(0,1fr) 220px; gap:18px; align-items:stretch; }}
    .summary p {{ margin:0; color:var(--muted); line-height:1.8; }}
    .signal-card {{ border:1px solid var(--line); border-radius:8px; padding:16px; background:#fbfcfd; }}
    .signal-card span {{ display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }}
    .signal-card strong {{ font-size:26px; color:var(--gold); }}
    main {{ padding:22px 0 42px; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:16px; }}
    .button {{ display:inline-flex; align-items:center; justify-content:center; min-height:38px; padding:0 14px; border-radius:7px; border:1px solid #2f6f9f; color:#2f6f9f; background:#fff; text-decoration:none; font-weight:800; }}
    section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:16px; }}
    h2 {{ margin:0 0 12px; font-size:20px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ color:var(--muted); font-size:12px; background:#f1f5f7; }}
    .badge {{ display:inline-flex; min-width:72px; justify-content:center; border-radius:999px; padding:5px 10px; font-weight:900; }}
    .badge.bullish {{ color:var(--red); background:var(--red-soft); }}
    .badge.bearish {{ color:var(--green); background:var(--green-soft); }}
    .badge.range,.badge.neutral {{ color:var(--gold); background:var(--gold-soft); }}
    .footer {{ color:var(--muted); font-size:12px; line-height:1.7; }}
    @media (max-width:760px) {{
      .summary {{ grid-template-columns:1fr; }}
      h1 {{ font-size:26px; }}
      table {{ display:block; overflow:auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="eyebrow">CreditBond AI</div>
      <h1>信用债 AI 日报</h1>
      <div class="summary">
        <p>{html.escape(s['plain'])}</p>
        <div class="signal-card"><span>综合方向</span><strong>{html.escape(s['direction'])}</strong></div>
      </div>
    </div>
  </header>
  <main class="wrap">
    <div class="actions">
      <a class="button" href="latest/report.html">打开完整 HTML 日报</a>
      {pdf_link}
      <a class="button" href="latest/report.json">查看 JSON</a>
    </div>
    <section>
      <h2>{html.escape(date_text)} 四期限信号</h2>
      <table>
        <thead><tr><th>期限</th><th>数据日</th><th>展示信号</th><th>严格信号</th><th>收益率</th><th>1日</th><th>5日</th><th>20日</th><th>看空</th><th>看多</th><th>震荡</th></tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </section>
    <p class="footer">本页由每日自动任务生成。展示口径用于阅读，不能替代人工风控和投资决策。</p>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    report = load_report(args.json_report)
    date_text = report_date(report) or "latest"
    out_dir = resolve_path(args.out_dir)
    html_report = resolve_path(args.html_report)
    json_report = resolve_path(args.json_report)
    pdf = resolve_path(args.pdf) if args.pdf else None

    latest_dir = out_dir / "latest"
    history_dir = out_dir / "history" / date_text
    copy_if_exists(html_report, latest_dir / "report.html")
    copy_if_exists(json_report, latest_dir / "report.json")
    copy_if_exists(html_report, history_dir / "report.html")
    copy_if_exists(json_report, history_dir / "report.json")
    has_pdf = bool(pdf and pdf.exists())
    if pdf and pdf.exists():
        copy_if_exists(pdf, latest_dir / "report.pdf")
        copy_if_exists(pdf, history_dir / "report.pdf")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(build_index(report, has_pdf), encoding="utf-8")
    (out_dir / "site_manifest.json").write_text(
        json.dumps(
            {
                "date": date_text,
                "latest_html": "latest/report.html",
                "latest_json": "latest/report.json",
                "latest_pdf": "latest/report.pdf" if has_pdf else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
