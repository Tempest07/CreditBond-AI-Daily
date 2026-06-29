from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from daily_report_utils import (
    ROOT,
    fmt_bp,
    fmt_prob,
    fmt_yield,
    load_report,
    report_date,
    resolve_path,
    summary,
    tenor_rows,
)


DEFAULT_JSON = "data/dm_daily_master_curve_2020/reports/daily_dm_update_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact PDF version of the credit-bond AI daily report.")
    parser.add_argument("--json-report", default=DEFAULT_JSON)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")),
        (Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/simhei.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("DailyCN", str(regular)))
            pdfmetrics.registerFont(TTFont("DailyCN-Bold", str(bold)))
            return "DailyCN", "DailyCN-Bold"
    return "Helvetica", "Helvetica-Bold"


def make_styles() -> dict[str, ParagraphStyle]:
    regular, bold = register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=bold, fontSize=20, leading=28, textColor=colors.HexColor("#18202a"), spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=13, leading=18, textColor=colors.HexColor("#18202a"), spaceBefore=8, spaceAfter=8),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=regular, fontSize=9.5, leading=15, textColor=colors.HexColor("#2b3440")),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName=regular, fontSize=8, leading=12, textColor=colors.HexColor("#5f6f80")),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontName=regular, fontSize=8, leading=10, textColor=colors.HexColor("#24303d")),
        "cell_bold": ParagraphStyle("cell_bold", parent=base["BodyText"], fontName=bold, fontSize=8, leading=10, textColor=colors.HexColor("#18202a")),
    }


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def build_pdf(report: dict, out_path: Path) -> Path:
    styles = make_styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=landscape(A4),
        rightMargin=1.25 * cm,
        leftMargin=1.25 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.15 * cm,
        title=f"信用债AI日报 {report_date(report)}",
    )

    rows = tenor_rows(report)
    s = summary(report)
    story = [
        p(f"信用债 AI 日报：{report_date(report)}", styles["title"]),
        p(f"综合方向：{s['direction']}", styles["h2"]),
        p(s["plain"], styles["body"]),
        Spacer(1, 0.25 * cm),
        p("期限信号", styles["h2"]),
    ]

    table_data = [[p(label, styles["cell_bold"]) for label in ["期限", "数据日", "展示信号", "严格信号", "最新收益率", "1日", "5日", "20日", "看空", "看多", "震荡"]]]
    for row in rows:
        table_data.append(
            [
                p(row["label"], styles["cell_bold"]),
                p(row["latest_date"], styles["cell"]),
                p(row["signal"], styles["cell_bold"]),
                p(row["strict_signal"], styles["cell"]),
                p(fmt_yield(row["latest_yield"]), styles["cell"]),
                p(fmt_bp(row["change_1d_bp"]), styles["cell"]),
                p(fmt_bp(row["change_5d_bp"]), styles["cell"]),
                p(fmt_bp(row["change_20d_bp"]), styles["cell"]),
                p(fmt_prob(row["prob_bearish"]), styles["cell"]),
                p(fmt_prob(row["prob_bullish"]), styles["cell"]),
                p(fmt_prob(row["prob_range"]), styles["cell"]),
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            1.25 * cm,
            2.25 * cm,
            2.0 * cm,
            1.75 * cm,
            1.85 * cm,
            1.35 * cm,
            1.35 * cm,
            1.35 * cm,
            1.3 * cm,
            1.3 * cm,
            1.3 * cm,
        ],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f6")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#dfe5ea")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    avg = s["avg_probabilities"]
    story.extend(
        [
            Spacer(1, 0.25 * cm),
            p("阅读口径", styles["h2"]),
            p("邮件与网页采用 40/60 展示口径：40% 至 60% 之间写作“震荡偏多/震荡偏空”，超过 60% 才写作强信号。严格信号仍保留原来的概率阈值与相对震荡优势过滤。", styles["body"]),
            p(f"四期限平均概率：看空 {fmt_prob(avg['看空'])}；看多 {fmt_prob(avg['看多'])}；震荡 {fmt_prob(avg['震荡'])}。", styles["body"]),
            p("本报告是研究和决策辅助材料，不构成自动交易指令。实际使用时仍需结合资金面、供给、信用事件和组合约束复核。", styles["small"]),
        ]
    )

    doc.build(story)
    return out_path


def main() -> int:
    args = parse_args()
    report = load_report(args.json_report)
    date_text = report_date(report) or "latest"
    out = resolve_path(args.out) if args.out else ROOT / "output" / "daily_reports" / f"creditbond_ai_daily_{date_text}.pdf"
    path = build_pdf(report, out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
