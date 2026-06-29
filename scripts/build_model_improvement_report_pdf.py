from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
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
TMP_DIR = ROOT / "tmp" / "pdfs" / "model_improvement"
PDF_PATH = OUT_DIR / "creditbond_ai_model_improvement_report.pdf"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
PLOT_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


@dataclass
class Experiment:
    key: str
    name: str
    method: str
    json_path: Path
    summary_path: Path | None = None
    note: str = ""
    primary: bool = True


EXPERIMENTS = [
    Experiment(
        key="baseline",
        name="旧基准：60日窗口、5日标签、三模型集成",
        method="基准",
        json_path=ROOT / "data/backtests/curve_2020_rolling_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "models/curve_2020_rolling_validation/rolling_validation_summary.csv",
        note="现有日报/回测主要参考口径。",
        primary=False,
    ),
    Experiment(
        key="derived",
        name="方法一：派生特征",
        method="过去变化、波动、期限利差等特征",
        json_path=ROOT / "data/backtests/experiments/derived_features_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "models/experiments/curve_2020_rolling_derived_features/rolling_validation_summary.csv",
        note="由于20日变化/波动特征需要历史，回测起点略晚。",
    ),
    Experiment(
        key="window120",
        name="方法二A：120日长窗口",
        method="把输入窗口从60日扩大到120日",
        json_path=ROOT / "data/backtests/experiments/window120_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "models/experiments/curve_2020_rolling_window120/rolling_validation_summary.csv",
        note="单独检验慢变量窗口。",
        primary=False,
    ),
    Experiment(
        key="window_ensemble",
        name="方法二：60日+120日双窗口集成",
        method="短窗口与长窗口概率平均",
        json_path=ROOT / "data/backtests/experiments/window60_120_ensemble_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "data/backtests/experiments/merged_summaries/window60_120_merged_summary.csv",
        note="与旧基准同为5日标签，可直接接入日报。",
    ),
    Experiment(
        key="h10",
        name="方法三：10日标签",
        method="预测未来10个交易日方向",
        json_path=ROOT / "data/backtests/experiments/h10_all_models_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "models/experiments/curve_2020_rolling_h10_all_models/rolling_validation_summary.csv",
        note="更像中期趋势信号，回测口径为用10日方向每日调仓。",
    ),
    Experiment(
        key="big",
        name="方法四：大模型",
        method="hidden=128、3层、8头、dropout=0.25",
        json_path=ROOT / "data/backtests/experiments/big_h5_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "models/experiments/curve_2020_rolling_big_h5/rolling_validation_summary.csv",
        note="明确测试4090可承载的更大模型。",
    ),
    Experiment(
        key="combo",
        name="额外验证：60日+120日+10日标签集成",
        method="短期、长窗口、中期标签概率平均",
        json_path=ROOT / "data/backtests/experiments/window60_120_h10_ensemble_portfolio/portfolio_backtest_report.json",
        summary_path=ROOT / "data/backtests/experiments/merged_summaries/window60_120_h10_merged_summary.csv",
        note="检验组合叠加是否继续提升。",
        primary=False,
    ),
]


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{digits}f}个百分点"


def load_report(exp: Experiment) -> dict:
    return json.loads(exp.json_path.read_text(encoding="utf-8"))


def load_metrics() -> pd.DataFrame:
    rows = []
    baseline = load_report(EXPERIMENTS[0])["portfolio_metrics"]
    for exp in EXPERIMENTS:
        report = load_report(exp)
        m = report["portfolio_metrics"]
        rows.append(
            {
                "key": exp.key,
                "name": exp.name,
                "method": exp.method,
                "start_date": m["start_date"],
                "end_date": m["end_date"],
                "periods": m["periods"],
                "total_return": m["total_return"],
                "annualized_return": m["annualized_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m.get("sharpe"),
                "positive_day_ratio": m.get("positive_day_ratio"),
                "average_position": m.get("average_position"),
                "buy_hold_total_return": m["buy_hold_total_return"],
                "neutral_total_return": m["neutral_total_return"],
                "excess_vs_baseline": m["total_return"] - baseline["total_return"],
                "html_report": report["html_path"],
                "json_report": str(exp.json_path.relative_to(ROOT)),
                "summary_path": str(exp.summary_path.relative_to(ROOT)) if exp.summary_path else "",
                "note": exp.note,
                "primary": exp.primary,
            }
        )
    return pd.DataFrame(rows)


def load_validation_summary(path: Path | None) -> dict[str, float | int | None]:
    if path is None or not path.exists():
        return {"rows": None, "macro_f1": None, "accuracy": None, "active_win": None}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        "rows": int(len(df)),
        "macro_f1": float(pd.to_numeric(df.get("macro_f1"), errors="coerce").mean()) if "macro_f1" in df else None,
        "accuracy": float(pd.to_numeric(df.get("accuracy"), errors="coerce").mean()) if "accuracy" in df else None,
        "active_win": float(pd.to_numeric(df.get("proxy_positive_active_ratio"), errors="coerce").mean())
        if "proxy_positive_active_ratio" in df
        else None,
    }


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MSYH", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", str(FONT_BOLD)))
    if PLOT_FONT.exists():
        font_manager.fontManager.addfont(str(PLOT_FONT))
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


def make_bar_chart(metrics: pd.DataFrame, path: Path) -> None:
    show = metrics[metrics["key"].isin(["baseline", "derived", "window_ensemble", "h10", "big", "combo"])].copy()
    labels = ["基准", "派生特征", "双窗口", "10日标签", "大模型", "组合验证"]
    values = show["total_return"].to_numpy() * 100
    colors_bar = ["#87919b", "#2f6f9f", "#238a62", "#1f78b4", "#c74343", "#b7812f"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, values, color=colors_bar, width=0.62)
    ax.axhline(values[0], color="#444", linewidth=1.1, linestyle="--", alpha=0.65, label="旧基准")
    ax.set_ylabel("累计收益率（%）")
    ax.set_title("四组提升实验累计收益率对比")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_line_chart(path: Path) -> None:
    curve_items = [
        ("基准", ROOT / "data/backtests/curve_2020_rolling_portfolio/portfolio_detail.csv", "#87919b"),
        ("双窗口", ROOT / "data/backtests/experiments/window60_120_ensemble_portfolio/portfolio_detail.csv", "#238a62"),
        ("10日标签", ROOT / "data/backtests/experiments/h10_all_models_portfolio/portfolio_detail.csv", "#1f78b4"),
        ("大模型", ROOT / "data/backtests/experiments/big_h5_portfolio/portfolio_detail.csv", "#c74343"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for label, csv_path, color in curve_items:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        ax.plot(pd.to_datetime(df["date"]), df["strategy_value"], label=label, color=color, linewidth=2.0)
    ax.set_title("主要方法净值走势")
    ax.set_ylabel("组合净值")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="MSYH-Bold",
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17202a"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="MSYH",
            fontSize=10,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#667789"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="MSYH-Bold",
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#17202a"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="MSYH-Bold",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#17202a"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=9.3,
            leading=15,
            textColor=colors.HexColor("#28323d"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#667789"),
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="MSYH-Bold",
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#17324d"),
            backColor=colors.HexColor("#eef5ff"),
            borderColor=colors.HexColor("#2f6f9f"),
            borderWidth=0.8,
            borderPadding=8,
            spaceAfter=10,
        ),
    }


def table_style(header="#f0f3f8") -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "MSYH-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "MSYH"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.4),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202a")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e0e8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def path_para(text: object, style: ParagraphStyle) -> Paragraph:
    value = escape(str(text))
    value = value.replace("\\", "\\<br/>").replace("/", "/<br/>")
    return Paragraph(value, style)


def build_pdf() -> Path:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    primary = metrics[metrics["primary"]].copy()

    bar_path = TMP_DIR / "experiment_bar.png"
    line_path = TMP_DIR / "net_value_lines.png"
    make_bar_chart(metrics, bar_path)
    make_line_chart(line_path)

    s = styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.35 * cm,
        leftMargin=1.35 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.2 * cm,
    )
    story = []
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph("信用债AI模型提升实验报告", s["title"]))
    story.append(Paragraph(f"生成时间：{generated}；数据：DM收益率曲线，2020年起；验证：滚动训练/滚动预测组合回测", s["subtitle"]))

    baseline = metrics[metrics["key"].eq("baseline")].iloc[0]
    best = primary.sort_values("total_return", ascending=False).iloc[0]
    window_best = metrics[metrics["key"].eq("window_ensemble")].iloc[0]
    big = metrics[metrics["key"].eq("big")].iloc[0]
    story.append(
        Paragraph(
            "结论：本轮最有效的单项方法是“10日标签”，累计收益率 "
            f"{pct(best['total_return'])}，相对旧基准提升 {pp(best['excess_vs_baseline'])}。"
            "更适合直接接入日报的是“双窗口集成”，因为它仍使用5日标签，信号含义与现有日报一致，"
            f"累计收益率 {pct(window_best['total_return'])}，最大回撤 {pct(window_best['max_drawdown'])}。"
            f"大模型实验累计收益率 {pct(big['total_return'])}，低于旧基准，说明当前瓶颈不是单纯模型容量。",
            s["callout"],
        )
    )

    story.append(Paragraph("一、实验总览", s["h1"]))
    rows = [["实验", "回测区间", "累计", "年化", "最大回撤", "夏普", "平均仓位", "相对基准"]]
    order = ["baseline", "derived", "window120", "window_ensemble", "h10", "big", "combo"]
    for _, row in metrics.set_index("key").loc[order].reset_index().iterrows():
        rows.append(
            [
                row["name"],
                f"{row['start_date']} 至 {row['end_date']}",
                pct(row["total_return"]),
                pct(row["annualized_return"]),
                pct(row["max_drawdown"]),
                "-" if pd.isna(row["sharpe"]) else f"{row['sharpe']:.2f}",
                pct(row["average_position"], 1),
                pp(row["excess_vs_baseline"]),
            ]
        )
    tbl = Table(rows, colWidths=[4.5 * cm, 3.15 * cm, 1.55 * cm, 1.45 * cm, 1.55 * cm, 1.15 * cm, 1.5 * cm, 2.0 * cm])
    tbl.setStyle(table_style())
    story.append(tbl)
    story.append(Paragraph("说明：派生特征和10日标签的样本期与基准略有差异，报告同时给出区间，避免误读。", s["small"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(bar_path), width=17.1 * cm, height=8.25 * cm))

    story.append(PageBreak())
    story.append(Paragraph("二、主要净值走势", s["h1"]))
    story.append(Paragraph("图中对比旧基准、双窗口、10日标签和大模型。可以看到10日标签在后段跟随债牛行情更充分；大模型没有提供稳定优势。", s["body"]))
    story.append(Image(str(line_path), width=17.1 * cm, height=8.65 * cm))

    story.append(Paragraph("三、四种方法的判断", s["h1"]))
    method_rows = [
        [para("方法", s["body"]), para("实际效果", s["body"]), para("判断", s["body"]), para("下一步", s["body"])],
        [
            para("派生特征", s["body"]),
            para("累计10.80%，略高于基准；年化3.10%。", s["body"]),
            para("有增量，但特征数量上升后过拟合和计算警告变多。", s["body"]),
            para("先做特征筛选和共线性压缩，再接入日报。", s["body"]),
        ],
        [
            para("双窗口集成", s["body"]),
            para("累计12.42%，最大回撤-2.29%，同口径优于基准。", s["body"]),
            para("最适合短期上线。它不是换目标，只是让模型同时看短周期和慢周期。", s["body"]),
            para("接入日报的主集成信号，并保留旧基准作对照。", s["body"]),
        ],
        [
            para("10日标签", s["body"]),
            para("累计13.88%，本轮最高；年化3.59%。", s["body"]),
            para("趋势噪声降低，收益改善明显。但信号含义从5日变成10日。", s["body"]),
            para("作为中期信号层接入，不直接替代5日信号；后续测试10日持仓规则。", s["body"]),
        ],
        [
            para("大模型", s["body"]),
            para("累计9.28%，低于基准10.69%。", s["body"]),
            para("当前历史长度下，大容量更容易记噪声，不是主要提升方向。", s["body"]),
            para("暂不默认启用；除非补更长历史或加入更强正则/预训练。", s["body"]),
        ],
    ]
    tbl = Table(method_rows, colWidths=[2.2 * cm, 3.9 * cm, 5.0 * cm, 5.8 * cm])
    tbl.setStyle(table_style("#edf6ef"))
    story.append(tbl)

    story.append(Paragraph("四、分类层指标", s["h1"]))
    val_rows = [["实验", "训练模型数", "平均准确率", "平均Macro F1", "主动方向胜率"]]
    for exp in EXPERIMENTS:
        if exp.key == "baseline" or exp.primary:
            v = load_validation_summary(exp.summary_path)
            val_rows.append(
                [
                    exp.name,
                    "-" if v["rows"] is None else str(v["rows"]),
                    "-" if v["accuracy"] is None else pct(v["accuracy"]),
                    "-" if v["macro_f1"] is None else pct(v["macro_f1"]),
                    "-" if v["active_win"] is None else pct(v["active_win"]),
                ]
            )
    tbl = Table(val_rows, colWidths=[6.2 * cm, 2.15 * cm, 2.3 * cm, 2.3 * cm, 2.45 * cm])
    tbl.setStyle(table_style())
    story.append(tbl)
    story.append(
        Paragraph(
            "解释：分类准确率和组合收益不是一一对应。信用债组合更关心看多/看空信号在关键波段是否有效，以及仓位路径是否能减少错误暴露。",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("五、建议的实施路线", s["h1"]))
    recommendations = [
        ("第一优先级", "把“60日+120日双窗口集成”接入日报，作为5日信号的主版本；旧基准保留一个月做并行监控。"),
        ("第二优先级", "把“10日标签”作为中期趋势模块展示，不与5日信号混成一个结论；日报中分别显示短期和中期方向。"),
        ("第三优先级", "派生特征继续做特征选择。先限制进入模型的派生特征数量，再滚动验证，避免特征膨胀。"),
        ("暂不推荐", "大模型不进入主线。当前数据只有2020年以来，样本量对128 hidden、3层结构偏少。"),
    ]
    rec_rows = [[para("优先级", s["body"]), para("动作", s["body"])]]
    rec_rows.extend([[para(a, s["body"]), para(b, s["body"])] for a, b in recommendations])
    tbl = Table(rec_rows, colWidths=[3.0 * cm, 13.6 * cm])
    tbl.setStyle(table_style("#eef5ff"))
    story.append(tbl)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("六、复现文件", s["h1"]))
    file_rows = [[para("实验", s["small"]), para("HTML报告", s["small"]), para("滚动汇总", s["small"])]]
    for _, row in metrics.iterrows():
        if row["key"] in ["baseline", "derived", "window_ensemble", "h10", "big", "combo"]:
            file_rows.append([para(row["name"], s["small"]), path_para(row["html_report"], s["small"]), path_para(row["summary_path"], s["small"])])
    tbl = Table(file_rows, colWidths=[4.0 * cm, 6.4 * cm, 6.2 * cm])
    tbl.setStyle(table_style())
    story.append(tbl)

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("MSYH", 7)
        canvas.setFillColor(colors.HexColor("#667789"))
        canvas.drawRightString(A4[0] - 1.35 * cm, 0.65 * cm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return PDF_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(path)
