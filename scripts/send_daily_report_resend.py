from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from daily_report_utils import load_report, report_date, resolve_path


RESEND_ENDPOINT = "https://api.resend.com/emails"
DEFAULT_JSON = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.json"
DEFAULT_HTML = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.html"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the full HTML credit-bond daily report through Resend.")
    parser.add_argument("--to", default=os.environ.get("RESEND_TO") or "yuweiqian@cib.com.cn")
    parser.add_argument("--from-email", default=os.environ.get("RESEND_FROM", ""))
    parser.add_argument("--subject", default="")
    parser.add_argument("--json-report", default=DEFAULT_JSON)
    parser.add_argument("--report", default=DEFAULT_HTML, help="Full HTML report used as the email body.")
    parser.add_argument("--pdf", default="", help="PDF report to attach.")
    parser.add_argument("--attach-html", action="store_true", help="Attach the full HTML report as a file as well.")
    parser.add_argument("--api-key-env", default="RESEND_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _attachment(path: Path) -> dict:
    return {
        "filename": path.name,
        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _inject_email_css(html: str) -> str:
    css = """
  <style id="creditbond-ai-email-overrides">
    body { margin: 0 !important; }
    table { border-collapse: collapse; }
    img, svg { max-width: 100%; }
    .wrap { width: min(1080px, calc(100% - 24px)) !important; }
  </style>
"""
    if "</head>" in html:
        return html.replace("</head>", f"{css}</head>", 1)
    return f"<!doctype html><html><head><meta charset=\"utf-8\">{css}</head><body>{html}</body></html>"


def _plain_text_from_html(html: str) -> str:
    no_style = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    parser = TextExtractor()
    parser.feed(no_style)
    lines = [line.strip() for line in parser.text().splitlines() if line.strip()]
    return "\n".join(lines[:120])


def build_payload(args: argparse.Namespace) -> dict:
    report_path = resolve_path(args.report)
    json_path = resolve_path(args.json_report)
    if not report_path.exists():
        raise FileNotFoundError(f"找不到日报 HTML：{report_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"找不到日报 JSON：{json_path}")
    if not args.from_email:
        raise ValueError("缺少发件地址：请设置 RESEND_FROM，或传入 --from-email。")

    report = load_report(json_path)
    date_text = report_date(report)
    report_html = _inject_email_css(report_path.read_text(encoding="utf-8"))
    plain_text = _plain_text_from_html(report_html)
    subject = args.subject or f"信用债AI日报 {date_text}"
    attachments = []
    if args.pdf:
        pdf_path = resolve_path(args.pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"找不到 PDF 日报：{pdf_path}")
        attachments.append(_attachment(pdf_path))
    if args.attach_html:
        attachments.append(_attachment(report_path))

    return {
        "from": args.from_email,
        "to": [args.to],
        "subject": subject,
        "text": plain_text,
        "html": report_html,
        "attachments": attachments,
        "tags": [
            {"name": "project", "value": "creditbond_ai"},
            {"name": "report_date", "value": date_text.replace("-", "_")},
        ],
    }


def send_with_resend(api_key: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        RESEND_ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "creditbond-ai-resend-client/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "response": json.loads(body) if body else {},
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return {"ok": False, "status": exc.code, "response": parsed}


def safe_payload(payload: dict) -> dict:
    safe = {k: v for k, v in payload.items() if k not in {"attachments", "html"}}
    html_body = str(payload.get("html", ""))
    safe["html"] = f"<html body hidden in dry-run; {len(html_body)} chars>"
    safe["attachments"] = [{"filename": item["filename"], "content": "<base64 hidden>"} for item in payload.get("attachments", [])]
    return safe


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    payload = build_payload(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "payload": safe_payload(payload)}, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"缺少 Resend API Key：请先设置环境变量 {args.api_key_env}。")
    result = send_with_resend(api_key, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
