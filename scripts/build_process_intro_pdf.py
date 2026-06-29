from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).absolute().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_PATH = OUT_DIR / "creditbond_ai_full_process_intro.pdf"

REPORT_JSON = ROOT / "data" / "dm_daily_master_smoke" / "reports" / "daily_dm_update_report.json"
BEST_BY_ENTITY_CSV = (
    ROOT / "data" / "experiments" / "no_data_strategy_sweep_full" / "strategy_sweep_best_by_entity.csv"
)
ROLLING_CSV = ROOT / "models" / "no_data_rolling_validation_v1" / "rolling_validation_summary.csv"


PALETTE = {
    "ink": colors.HexColor("#17202E"),
    "muted": colors.HexColor("#58677A"),
    "line": colors.HexColor("#D9E1EA"),
    "paper": colors.HexColor("#F7F9FC"),
    "blue": colors.HexColor("#2F6FA5"),
    "teal": colors.HexColor("#27896B"),
    "amber": colors.HexColor("#B98222"),
    "red": colors.HexColor("#C94346"),
    "lavender": colors.HexColor("#6E63B6"),
}


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    bold_candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    regular = next(p for p in regular_candidates if p.exists())
    bold = next(p for p in bold_candidates if p.exists())
    pdfmetrics.registerFont(TTFont("CJK", str(regular)))
    pdfmetrics.registerFont(TTFont("CJKBold", str(bold)))
    return "CJK", "CJKBold"


FONT, FONT_BOLD = register_fonts()


def pstyle(name: str, **kwargs) -> ParagraphStyle:
    base = {
        "fontName": FONT,
        "fontSize": 10.5,
        "leading": 16,
        "textColor": PALETTE["ink"],
        "wordWrap": "CJK",
        "alignment": TA_LEFT,
        "spaceAfter": 6,
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


styles = getSampleStyleSheet()
S = {
    "cover_title": pstyle(
        "cover_title",
        fontName=FONT_BOLD,
        fontSize=30,
        leading=38,
        textColor=colors.white,
        alignment=TA_LEFT,
        spaceAfter=14,
    ),
    "cover_sub": pstyle(
        "cover_sub",
        fontSize=13.5,
        leading=21,
        textColor=colors.HexColor("#E9F1FA"),
        spaceAfter=14,
    ),
    "title": pstyle(
        "title",
        fontName=FONT_BOLD,
        fontSize=21,
        leading=28,
        textColor=PALETTE["ink"],
        spaceAfter=10,
    ),
    "subtitle": pstyle(
        "subtitle",
        fontSize=12,
        leading=18,
        textColor=PALETTE["muted"],
        spaceAfter=12,
    ),
    "h2": pstyle(
        "h2",
        fontName=FONT_BOLD,
        fontSize=14,
        leading=20,
        textColor=PALETTE["ink"],
        spaceBefore=4,
        spaceAfter=7,
    ),
    "body": pstyle("body", fontSize=10.8, leading=17, spaceAfter=6),
    "small": pstyle("small", fontSize=8.7, leading=12.5, textColor=PALETTE["muted"], spaceAfter=3),
    "table": pstyle("table", fontSize=8.8, leading=12.5, spaceAfter=0),
    "table_bold": pstyle("table_bold", fontName=FONT_BOLD, fontSize=8.8, leading=12.5, spaceAfter=0),
    "table_header": pstyle(
        "table_header",
        fontName=FONT_BOLD,
        fontSize=8.8,
        leading=12.5,
        textColor=colors.white,
        spaceAfter=0,
    ),
    "metric": pstyle("metric", fontName=FONT_BOLD, fontSize=14.5, leading=18.5, spaceAfter=2),
    "label": pstyle("label", fontSize=8.8, leading=12, textColor=PALETTE["muted"], spaceAfter=1),
    "note": pstyle(
        "note",
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor("#25415C"),
        backColor=colors.HexColor("#EEF5FA"),
        borderColor=colors.HexColor("#D5E5F1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8,
    ),
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "-"


def bp(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):+.{digits}f} bp"
    except Exception:
        return "-"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_model_name(model_dir: str) -> str:
    lower = model_dir.replace("\\", "/").lower()
    if lower.endswith("/gru"):
        return "GRU"
    if lower.endswith("/tcn"):
        return "TCN"
    if lower.endswith("/transformer"):
        return "Transformer"
    return Path(lower).name.upper()


def load_state() -> dict:
    report = read_json(REPORT_JSON)
    tenors = []
    for item in report.get("market_snapshot", {}).get("tenors", []):
        mp = item.get("model_prediction") or {}
        tenors.append(
            {
                "期限": item.get("label", "-"),
                "数据日期": item.get("latest_date", report.get("end_date", "-")),
                "收益率": f"{item.get('latest_yield', 0):.4f}%" if item.get("latest_yield") is not None else "-",
                "1日变化": bp(item.get("change_1d_bp")),
                "5日变化": bp(item.get("change_5d_bp")),
                "20日变化": bp(item.get("change_20d_bp")),
                "主力信号": mp.get("prediction", "-"),
                "模型": infer_model_name(mp.get("model_dir", "")),
            }
        )

    model_dirs = [p.get("model_dir", "") for p in report.get("predictions", []) if p.get("model_dir")]
    model_counts = defaultdict(int)
    for path in model_dirs:
        model_counts[infer_model_name(path)] += 1

    best_rows = read_csv(BEST_BY_ENTITY_CSV)
    rolling_rows = read_csv(ROLLING_CSV)

    best_by_tenor: dict[str, dict] = {}
    for row in best_rows:
        tenor = row.get("tenor", "")
        current = best_by_tenor.get(tenor)
        if current is None or float(row.get("total_return", 0) or 0) > float(
            current.get("total_return", 0) or 0
        ):
            best_by_tenor[tenor] = row

    rolling_summary = []
    by_tenor = defaultdict(list)
    for row in rolling_rows:
        by_tenor[row.get("tenor", "-")].append(row)
    for tenor, rows in sorted(by_tenor.items(), key=lambda x: float(x[0]) if x[0] else 999):
        def avg(key: str) -> float:
            vals = [float(r.get(key, 0) or 0) for r in rows]
            return sum(vals) / len(vals) if vals else 0

        positive = sum(1 for r in rows if float(r.get("proxy_total_return", 0) or 0) > 0)
        rolling_summary.append(
            {
                "期限": f"{tenor}年",
                "平均准确率": pct(avg("accuracy"), 1),
                "平均Macro F1": pct(avg("macro_f1"), 1),
                "正收益折数": f"{positive}/{len(rows)}",
            }
        )

    return {
        "run_date": report.get("run_date") or str(date.today()),
        "data_start": report.get("data_start", "-"),
        "data_end": report.get("data_end", "-"),
        "feature_shape": report.get("features_shape", ["-", "-"]),
        "tenors": tenors,
        "model_dirs": model_dirs,
        "model_counts": dict(model_counts),
        "best_by_tenor": best_by_tenor,
        "rolling_summary": rolling_summary,
    }


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(lines: list[str]) -> list[Paragraph]:
    return [para(f"• {line}", "body") for line in lines]


def table(data: list[list[object]], widths: list[float] | None = None, header: bool = True) -> Table:
    wrapped = []
    for r, row in enumerate(data):
        wrapped.append(
            [
                cell
                if isinstance(cell, Flowable)
                else para(esc(cell), "table_header" if header and r == 0 else "table")
                for cell in row
            ]
        )
    t = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, PALETTE["line"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), PALETTE["ink"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for r in range(1 if header else 0, len(data)):
        if r % 2 == 1:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FBFCFE")))
    t.setStyle(TableStyle(style))
    return t


def metric_card(label: str, value: str, note: str, accent) -> Table:
    content = [
        [para(esc(label), "label")],
        [para(esc(value), "metric")],
        [para(esc(note), "small")],
    ]
    t = Table(content, colWidths=[48 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#DCE5EE")),
                ("LINEABOVE", (0, 0), (-1, 0), 4, accent),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def flow_grid(items: list[tuple[str, str, object]]) -> Table:
    rows = []
    for i in range(0, len(items), 2):
        pair = items[i : i + 2]
        row = []
        for title, body, accent in pair:
            cell = Table(
                [[para(esc(title), "table_bold")], [para(esc(body), "table")]],
                colWidths=[77 * mm],
            )
            cell.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#DCE5EE")),
                        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            row.append(cell)
        if len(row) == 1:
            row.append("")
        rows.append(row)
    t = Table(rows, colWidths=[82 * mm, 82 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def cover_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(PALETTE["ink"])
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["blue"])
    canvas.rect(0, h - 105 * mm, w, 105 * mm, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["teal"])
    canvas.rect(0, h - 108 * mm, w * 0.58, 3 * mm, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["amber"])
    canvas.rect(w * 0.58, h - 108 * mm, w * 0.22, 3 * mm, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["red"])
    canvas.rect(w * 0.80, h - 108 * mm, w * 0.20, 3 * mm, fill=1, stroke=0)
    canvas.restoreState()


def later_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(PALETTE["line"])
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, h - 18 * mm, w - 18 * mm, h - 18 * mm)
    canvas.setFillColor(PALETTE["muted"])
    canvas.setFont(FONT, 8)
    canvas.drawString(18 * mm, 11 * mm, "CreditBond AI · 研究和决策辅助框架")
    canvas.drawRightString(w - 18 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def add_section(story: list, title: str, subtitle: str | None = None):
    story.append(para(title, "title"))
    if subtitle:
        story.append(para(subtitle, "subtitle"))


def build_pdf() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=24 * mm,
        bottomMargin=19 * mm,
        title="信用债 AI 研究与决策辅助框架全流程简介",
        author="CreditBond AI",
    )
    story: list = []

    story.append(Spacer(1, 28 * mm))
    story.append(para("信用债 AI 研究与决策辅助框架", "cover_title"))
    story.append(
        para(
            "面向非 AI 观众的全流程简介：数据如何进入系统，神经网络如何学习，预测如何变成日报、回测和投资研究辅助。",
            "cover_sub",
        )
    )
    story.append(Spacer(1, 10 * mm))
    cover_info = Table(
        [
            [para("定位", "table_bold"), para("本系统不是自动下单机器，而是把信用债研究流程结构化、可回测、可复盘的 AI 助手。", "table")],
            [para("当前日期", "table_bold"), para(state["run_date"], "table")],
            [para("当前覆盖", "table_bold"), para("3年、5年、10年、20年信用债期限；每个期限独立输出信号。", "table")],
        ],
        colWidths=[28 * mm, 128 * mm],
        hAlign="LEFT",
    )
    cover_info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D6E0EA")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6E0EA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(cover_info)
    story.append(PageBreak())

    add_section(story, "一页讲清楚", "先把故事压缩成普通人能抓住的版本。")
    story.append(
        flow_grid(
            [
                ("1. 数据", "从 DM API 自动拉取经济、利率、资金面、信用债收益率等指标。", PALETTE["blue"]),
                ("2. 清洗", "按日期对齐；缺失值只允许向前填充；记录指标可获得时间，防止未来数据混入。", PALETTE["teal"]),
                ("3. 特征", "把原始指标加工成神经网络可读的时间序列，例如变化、差值、滚动统计。", PALETTE["amber"]),
                ("4. 模型", "本地 RTX 4090 训练 GRU、TCN、Transformer，不靠手写涨跌规则。", PALETTE["lavender"]),
                ("5. 信号", "每个期限输出看空、看多、震荡三类概率，再做同期限模型集成和阈值过滤。", PALETTE["red"]),
                ("6. 报告", "生成日报、回测和实验报告，供研究员解释、比较、复盘和决策。", PALETTE["blue"]),
            ]
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(
        para(
            "一句话：系统先把数据整理得足够干净，再让神经网络学习历史中“什么环境下信用债收益率更可能上行、下行或震荡”，最后把概率翻译成可读的研究信号。",
            "note",
        )
    )
    story.append(PageBreak())

    add_section(story, "为什么需要这个系统", "信用债判断不是看单一数字，而是同时看期限、评级、利率、资金面、宏观和历史位置。")
    story.extend(
        bullet(
            [
                "人工研究容易被当天市场噪声影响；系统强制每天用同一套数据和同一套规则复盘。",
                "传统表格只能展示发生了什么；神经网络试图学习“这些变化组合起来，未来几天可能意味着什么”。",
                "信用债投资最怕无纪律地追涨杀跌；概率、阈值和回测可以让信号更透明。",
                "最终判断仍然由人完成，AI 负责提供结构化证据和一套不会忘记历史的参照系。",
            ]
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Table(
            [
                [
                    metric_card("数据区间", f"{state['data_start']} 至 {state['data_end']}", "当前日报使用的数据覆盖范围", PALETTE["blue"]),
                    metric_card("模型数量", f"{len(state['model_dirs'])} 个", "3/5/10/20 年 × GRU/TCN/Transformer", PALETTE["lavender"]),
                    metric_card("特征表规模", f"{state['feature_shape'][0]} 行 × {state['feature_shape'][1]} 列", "进入模型前的宽表/特征表", PALETTE["teal"]),
                ]
            ],
            colWidths=[55 * mm, 55 * mm, 55 * mm],
        )
    )
    story.append(PageBreak())

    add_section(story, "数据底座", "这一步比模型更重要：数据错了，AI 会学得很努力，但方向会错。")
    story.append(
        table(
            [
                ["环节", "做什么", "为什么重要"],
                ["DM API 抓取", "按配置表批量拉取 EDB 和收益率曲线数据。", "减少手工复制粘贴，保证每天流程一致。"],
                ["数据字典", "记录指标名、代码、频率、可获得时间、所属类别。", "以后一次性补很多指标时，能知道每列是什么。"],
                ["只向前填充", "整理数据时只允许用历史值补今天，不能用未来值补过去。", "避免“未来函数”，这是金融 AI 回测最常见的坑。"],
                ["模型就绪表", "把不同频率数据对齐到同一张时间表。", "让日频、月频、季频指标可以一起被模型读取。"],
                ["审计报告", "记录缺失率、最早日期、最晚日期、被剔除字段。", "先知道数据质量，再决定是否入模。"],
            ],
            widths=[30 * mm, 69 * mm, 68 * mm],
        )
    )
    story.append(PageBreak())

    add_section(story, "神经网络层", "当前不是单一模型押注，而是同一问题用多种神经网络同时观察。")
    story.append(
        table(
            [
                ["模型", "普通解释", "适合解决的问题"],
                ["GRU 门控循环神经网络", "像一个会保留近期记忆的读表器，逐日读历史数据。", "样本不算特别多时较稳，训练速度快。"],
                ["TCN 时序卷积网络", "像同时拿不同长度的尺子量历史走势。", "并行度高，比较适合 RTX 4090。"],
                ["Transformer 注意力模型", "会判断哪几段历史更值得关注。", "擅长复杂关系，但更吃数据，必须防过拟合。"],
            ],
            widths=[42 * mm, 70 * mm, 55 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        para(
            "目前四个期限各自训练专属模型。也就是说，3 年模型只学习 3 年信用债未来几天的方向，5 年、10 年、20 年同理；它们不会被合成一个“统一大信号”。",
            "note",
        )
    )
    model_count_rows = [["模型结构", "当前数量"]]
    for name in ["GRU", "TCN", "Transformer"]:
        model_count_rows.append([name, str(state["model_counts"].get(name, 0))])
    story.append(table(model_count_rows, widths=[60 * mm, 35 * mm]))
    story.append(PageBreak())

    add_section(story, "信号怎样产生", "AI 输出的不是一句“买/卖”，而是三种状态的概率。")
    story.append(
        table(
            [
                ["输出", "含义", "对信用债价格的直观理解"],
                ["看空", "未来目标收益率上行概率较高。", "收益率上行通常对应价格承压，应降低风险暴露或等待买点。"],
                ["看多", "未来目标收益率下行概率较高。", "收益率下行通常对应价格受益，可提高或维持积极持仓。"],
                ["震荡", "未来变化不够明显，或者模型分歧较大。", "不急着动，优先观察和复盘。"],
            ],
            widths=[30 * mm, 68 * mm, 69 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.extend(
        bullet(
            [
                "同一期限下，GRU、TCN、Transformer 的概率会被平均，形成“集成信号”。",
                "如果方向概率没有超过阈值，或者相对震荡优势不够，系统会把信号降级为震荡/信号弱。",
                "这样做的目的不是追求每天都有动作，而是减少低置信度预测对研究判断的干扰。",
            ]
        )
    )
    story.append(PageBreak())

    add_section(story, "当前日报怎么读", "日报是给普通读者看的，不只是 Excel 里的数字。")
    tenor_rows = [["期限", "数据日期", "最新收益率", "1日变化", "5日变化", "20日变化", "主力模型信号"]]
    for row in state["tenors"]:
        tenor_rows.append(
            [
                row["期限"],
                row["数据日期"],
                row["收益率"],
                row["1日变化"],
                row["5日变化"],
                row["20日变化"],
                row["主力信号"],
            ]
        )
    story.append(table(tenor_rows, widths=[18 * mm, 27 * mm, 25 * mm, 24 * mm, 24 * mm, 24 * mm, 25 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(
        para(
            "日报里四个期限会按同等地位展示：每个期限都有走势小图、收益率变化、信用利差、集成概率、阈值后信号和模型明细。主力模型只是参照之一，不代表系统只相信一个模型。",
            "note",
        )
    )
    story.append(PageBreak())

    add_section(story, "回测与实验", "回测不是证明未来一定赚钱，而是检查这套信号过去有没有纪律性。")
    best_rows = [["期限", "当前最佳回测累计收益", "最大回撤", "交易次数", "提示"]]
    for tenor in ["3", "5", "10", "20"]:
        row = state["best_by_tenor"].get(tenor, {})
        hint = {
            "3": "可改善，但滚动稳定性仍不足",
            "5": "当前较弱，需要更好特征或更长历史",
            "10": "滚动验证目前相对最稳",
            "20": "策略表现强，但分类稳定性需继续验证",
        }.get(tenor, "")
        best_rows.append(
            [
                f"{tenor}年",
                pct(row.get("total_return"), 2),
                pct(row.get("max_drawdown"), 2),
                row.get("trade_count", "-"),
                hint,
            ]
        )
    story.append(table(best_rows, widths=[18 * mm, 37 * mm, 30 * mm, 24 * mm, 58 * mm]))
    story.append(Spacer(1, 5 * mm))
    rolling_rows = [["期限", "滚动平均准确率", "滚动平均 Macro F1", "正收益折数"]]
    for row in state["rolling_summary"]:
        rolling_rows.append([row["期限"], row["平均准确率"], row["平均Macro F1"], row["正收益折数"]])
    story.append(table(rolling_rows, widths=[35 * mm, 45 * mm, 45 * mm, 40 * mm]))
    story.append(PageBreak())

    add_section(story, "现在最重要的限制", "模型不准时，不一定是“神经网络不行”，也可能是数据历史和标签设计还不够。")
    story.extend(
        bullet(
            [
                "当前 CFETS 相关日频数据历史偏短，神经网络能学习到的市场状态还不够多。",
                "信用债未来 5 天方向本身噪声很大，单纯追求分类准确率会误导判断。",
                "回测仍是 proxy 版本，需要继续加入交易成本、久期、carry、流动性和组合层约束。",
                "概率还需要校准：模型说 60% 时，真实历史上是否真的接近 60%，要单独检验。",
                "后续补数据时，要优先提升历史长度和可获得时间质量，而不是盲目堆很多相关性不明的指标。",
            ]
        )
    )
    story.append(PageBreak())

    add_section(story, "后续路线图", "目标不是做一个神秘黑箱，而是做一套能每天迭代、能被人理解的研究机器。")
    story.append(
        flow_grid(
            [
                ("第一阶段：数据质量", "补齐 CFETS 历史、数据字典、发布时间、缺失审计。", PALETTE["teal"]),
                ("第二阶段：模型稳健性", "滚动训练、概率校准、模型集成、不同标签阈值实验。", PALETTE["lavender"]),
                ("第三阶段：交易解释", "把方向信号接入久期、carry、利差、交易成本与仓位规则。", PALETTE["amber"]),
                ("第四阶段：研究闭环", "日报、回测、误判复盘、人工备注、下次训练自动吸收。", PALETTE["blue"]),
            ]
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(
        para(
            "理想状态下，这套系统每天先自动拉数据、更新特征、跑模型、生成报告；研究员再判断模型为什么这么看、哪些信号可信、哪些信号需要忽略。",
            "note",
        )
    )
    story.append(PageBreak())

    add_section(story, "给普通观众的结尾", "可以这样介绍这个项目。")
    story.append(
        para(
            "我们做的不是一个替人拍脑袋买卖债券的机器人，而是一套信用债研究辅助系统。它每天从数据库拿数据，先保证数据没有偷看未来，再用本地神经网络学习不同市场环境下未来几天收益率更可能上行、下行还是震荡。最后，系统把结果整理成每个期限独立的概率、信号、回测和图表。人的角色没有被替代：人负责理解宏观和信用环境，AI 负责把历史经验、纪律和重复计算稳定地摆在桌面上。",
            "body",
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        table(
            [
                ["使用边界", "这是研究和决策辅助框架，不构成投资建议或收益承诺。正式使用前仍需风控、组合约束和人工复核。"],
                ["核心原则", "宁愿少动，也不要让不确定的模型信号驱动仓位；宁愿先审计数据，也不要让漂亮回测掩盖未来函数。"],
            ],
            widths=[30 * mm, 137 * mm],
            header=False,
        )
    )

    doc.build(story, onFirstPage=cover_page, onLaterPages=later_page)
    return OUT_PATH


if __name__ == "__main__":
    print(build_pdf())
