from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
import urllib.error
import urllib.request
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


RESEND_ENDPOINT = "https://api.resend.com/emails"
DEFAULT_JSON = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.json"
DEFAULT_HTML = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the credit-bond daily report through Resend.")
    parser.add_argument("--to", default=os.environ.get("RESEND_TO") or "yuweiqian@cib.com.cn")
    parser.add_argument("--from-email", default=os.environ.get("RESEND_FROM", ""))
    parser.add_argument("--subject", default="")
    parser.add_argument("--json-report", default=DEFAULT_JSON)
    parser.add_argument(
        "--report",
        default=DEFAULT_HTML,
        help="HTML report. It is linked in dry-run output and can be attached with --attach-html.",
    )
    parser.add_argument("--pdf", default="", help="PDF report to attach.")
    parser.add_argument("--attach-html", action="store_true", help="Attach the full HTML report as well.")
    parser.add_argument("--api-key-env", default="RESEND_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _attachment(path: Path) -> dict:
    return {
        "filename": path.name,
        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _badge_style(signal: str) -> str:
    tone = signal_tone(signal)
    if tone == "bullish":
        return "color:#c74343;background:#f8dddd;"
    if tone == "bearish":
        return "color:#238a62;background:#dff2e9;"
    return "color:#b7812f;background:#f8edda;"


def build_email_body(report: dict) -> tuple[str, str]:
    date_text = report_date(report)
    s = summary(report)
    rows = tenor_rows(report)
    plain_lines = [
        f"信用债 AI 日报：{date_text}",
        f"综合方向：{s['direction']}",
        s["plain"],
        "",
        "四期限信号：",
    ]
    table_rows = []
    for row in rows:
        plain_lines.append(
            f"- {row['label']}：{row['signal']}；严格信号 {row['strict_signal']}；"
            f"收益率 {fmt_yield(row['latest_yield'])}；1日 {fmt_bp(row['change_1d_bp'])}；"
            f"5日 {fmt_bp(row['change_5d_bp'])}；20日 {fmt_bp(row['change_20d_bp'])}。"
        )
        table_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['label'])}</td>
              <td>{html.escape(row['latest_date'])}</td>
              <td><span style="{_badge_style(row['signal'])}border-radius:999px;padding:4px 10px;font-weight:700;white-space:nowrap;">{html.escape(row['signal'])}</span></td>
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

    html_body = f"""<!doctype html>
<html lang="zh-CN">
<body style="margin:0;background:#f5f7f8;color:#17202a;font-family:Microsoft YaHei,Segoe UI,Arial,sans-serif;">
  <div style="max-width:920px;margin:0 auto;padding:24px 16px;">
    <div style="background:#ffffff;border:1px solid #dde5ec;border-radius:10px;padding:22px;">
      <div style="color:#2f6f9f;font-weight:800;margin-bottom:8px;">CreditBond AI</div>
      <h1 style="margin:0 0 12px;font-size:26px;line-height:1.25;">信用债 AI 日报：{html.escape(date_text)}</h1>
      <div style="display:inline-block;border:1px solid #dde5ec;border-radius:8px;padding:12px 16px;margin:4px 0 14px;background:#fbfcfd;">
        <div style="font-size:13px;color:#667789;">综合方向</div>
        <div style="font-size:24px;font-weight:900;color:#b7812f;">{html.escape(s['direction'])}</div>
      </div>
      <p style="margin:0 0 18px;color:#526273;line-height:1.75;">{html.escape(s['plain'])}</p>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#f1f5f7;color:#667789;">
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">期限</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">数据日</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">展示信号</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">严格信号</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">收益率</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">1日</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">5日</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">20日</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">看空</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">看多</th>
            <th style="text-align:left;padding:9px;border-bottom:1px solid #dde5ec;">震荡</th>
          </tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
      <p style="margin:18px 0 0;color:#667789;font-size:12px;line-height:1.7;">PDF 日报见附件。本邮件是研究和决策辅助材料，不构成自动交易指令。</p>
    </div>
  </div>
</body>
</html>"""
    return "\n".join(plain_lines), html_body


def build_payload(args: argparse.Namespace) -> dict:
    report_path = resolve_path(args.report)
    json_path = resolve_path(args.json_report)
    if not report_path.exists():
        raise FileNotFoundError(f"找不到日报文件：{report_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"找不到日报 JSON：{json_path}")
    if not args.from_email:
        raise ValueError("缺少发件地址：请设置 RESEND_FROM，或传入 --from-email。")

    report = load_report(json_path)
    plain_text, body_html = build_email_body(report)
    subject = args.subject or f"信用债AI日报 {report_date(report)}"
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
        "html": body_html,
        "attachments": attachments,
        "tags": [
            {"name": "project", "value": "creditbond_ai"},
            {"name": "report_date", "value": report_date(report).replace("-", "_")},
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
    safe = {k: v for k, v in payload.items() if k != "attachments"}
    safe["attachments"] = [{"filename": item["filename"], "content": "<base64 hidden>"} for item in payload.get("attachments", [])]
    return safe


def main() -> int:
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
