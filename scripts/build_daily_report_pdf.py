from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from daily_report_utils import ROOT, report_date, resolve_path


DEFAULT_JSON = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.json"
DEFAULT_HTML = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.html"


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the full HTML daily report to PDF.")
    parser.add_argument("--json-report", default=DEFAULT_JSON, help="Used for report date fallback.")
    parser.add_argument("--html-report", default=DEFAULT_HTML, help="Full HTML daily report to render.")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def read_report_date(json_report: Path) -> str:
    if not json_report.exists():
        return "latest"
    try:
        report = json.loads(json_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "latest"
    return report_date(report) or "latest"


def find_browser() -> Path | None:
    env_candidates = [os.environ.get("CHROME_PATH"), os.environ.get("EDGE_PATH")]
    path_candidates = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in env_candidates + path_candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def make_print_html(source_html: Path, target_html: Path) -> None:
    html = source_html.read_text(encoding="utf-8")
    print_css = """
  <style id="creditbond-ai-pdf-print">
    @page { size: A4; margin: 8mm; }
    html, body {
      background: #ffffff !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    .wrap { width: 100% !important; max-width: none !important; }
    .top { border-bottom: 1px solid #dfe5ea !important; }
    .hero {
      padding: 10px 0 9px !important;
      display: grid !important;
      grid-template-columns: minmax(0, 1fr) 210px !important;
      gap: 12px !important;
      align-items: stretch !important;
    }
    h1 { font-size: 23px !important; line-height: 1.18 !important; }
    .eyebrow { font-size: 11px !important; margin-bottom: 4px !important; }
    .hero p { font-size: 11.5px !important; line-height: 1.45 !important; margin-top: 6px !important; }
    .signal-box { padding: 10px 12px !important; }
    .signal-label { font-size: 10.5px !important; }
    .signal-value { font-size: 22px !important; }
    .muted { font-size: 11px !important; line-height: 1.45 !important; }
    .main { padding: 10px 0 0 !important; gap: 8px !important; }
    .section { padding: 11px !important; border-radius: 6px !important; }
    .section h2 { font-size: 14px !important; margin-bottom: 8px !important; }
    .takeaway { font-size: 11.5px !important; line-height: 1.5 !important; }
    .meta-grid { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; gap: 7px !important; margin-top: 9px !important; }
    .metric { padding: 7px 8px !important; border-top-width: 2px !important; }
    .metric span { font-size: 9.5px !important; }
    .metric strong { font-size: 12px !important; }
    .term-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 8px !important; }
    .model-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 8px !important; }
    .history-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 8px !important; }
    .term-card, .model-block { padding: 10px !important; border-radius: 6px !important; }
    .term-head h3 { font-size: 15px !important; }
    .model-head h3 { font-size: 12px !important; }
    .pill { font-size: 10px !important; min-width: 44px !important; padding: 3px 7px !important; }
    .spark-wrap { margin: 7px 0 !important; }
    .spark-wrap svg { height: 92px !important; }
    .spark-dates { font-size: 9.5px !important; }
    .term-metrics { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; gap: 5px !important; }
    .term-metrics div { padding: 6px !important; }
    .term-metrics span { font-size: 9px !important; }
    .term-metrics strong { font-size: 10.5px !important; }
    .spread-line, .ensemble-probs { margin-top: 7px !important; padding: 7px !important; font-size: 10.5px !important; }
    .prob-list { gap: 6px !important; }
    .prob-row { grid-template-columns: 34px 1fr 42px !important; gap: 6px !important; font-size: 10.5px !important; }
    .prob-track { height: 7px !important; }
    .history-card { padding: 8px !important; font-size: 10.5px !important; }
    .history-card strong { font-size: 11.5px !important; }
    .note, .file-list { font-size: 10.5px !important; line-height: 1.5 !important; }
    .term-card, .model-block, .metric, .history-row {
      break-inside: avoid;
      page-break-inside: avoid;
    }
  </style>
"""
    if "</head>" in html:
        html = html.replace("</head>", f"{print_css}</head>", 1)
    else:
        html = f"<!doctype html><html><head>{print_css}</head><body>{html}</body></html>"
    target_html.write_text(html, encoding="utf-8")


def render_pdf_with_browser(html_report: Path, out_path: Path) -> Path:
    browser = find_browser()
    if browser is None:
        raise RuntimeError("找不到 Chrome 或 Edge，无法把完整 HTML 日报渲染为 PDF。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="creditbond_ai_pdf_") as tmp:
        tmp_dir = Path(tmp)
        print_html = tmp_dir / "daily_report_print.html"
        profile_dir = tmp_dir / "browser_profile"
        make_print_html(html_report, print_html)
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile_dir}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={out_path}",
            print_html.as_uri(),
        ]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                "HTML 转 PDF 失败："
                + (result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}")
            )

    if not out_path.exists() or out_path.stat().st_size < 10_000:
        raise RuntimeError(f"PDF 未正常生成或文件过小：{out_path}")
    return out_path


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    html_report = resolve_path(args.html_report)
    json_report = resolve_path(args.json_report)
    if not html_report.exists():
        raise FileNotFoundError(f"找不到 HTML 日报：{html_report}")
    date_text = read_report_date(json_report)
    out = resolve_path(args.out) if args.out else ROOT / "output" / "daily_reports" / f"creditbond_ai_daily_{date_text}.pdf"
    path = render_pdf_with_browser(html_report, out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
