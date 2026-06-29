from __future__ import annotations

import ast
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
from reportlab.lib.pagesizes import A4, landscape
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
TMP_DIR = ROOT / "tmp" / "pdfs" / "model_improvement_detailed"
PDF_PATH = OUT_DIR / "creditbond_ai_model_improvement_report_detailed.pdf"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
PLOT_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


@dataclass(frozen=True)
class Experiment:
    key: str
    short_name: str
    full_name: str
    hypothesis: str
    change: str
    command: str
    json_path: Path
    portfolio_summary_path: Path
    validation_summary_path: Path | None
    conclusion: str
    action: str
    caveat: str


EXPERIMENTS = [
    Experiment(
        key="baseline",
        short_name="旧基准",
        full_name="旧基准：60日窗口 + 5日标签 + 三模型集成",
        hypothesis="作为对照组，不做新增数据、不改标签、不改模型容量；用于衡量后续实验是否真的带来增量。",
        change="每个期限训练 GRU、TCN、Transformer 三类时序神经网络；输入最近60个交易日；预测未来5个交易日方向。",
        command="python scripts/run_curve_2020_rolling_validation.py --out-dir models/curve_2020_rolling_validation --window 60 --horizon 5 --models gru,tcn,transformer --device cuda",
        json_path=ROOT / "data/backtests/curve_2020_rolling_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/curve_2020_rolling_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "models/curve_2020_rolling_validation/rolling_validation_summary.csv",
        conclusion="老框架不是无效，回测已经能在不扣成本口径下获得正收益，但它对牛市整体趋势的利用不充分。",
        action="继续作为日报对照组保留。",
        caveat="这是所有实验的基准，不应把它当成最优模型。",
    ),
    Experiment(
        key="derived",
        short_name="派生特征",
        full_name="方法一：派生特征",
        hypothesis="不增加外部数据，只从已有曲线和经济指标中提取更容易被神经网络识别的形态，例如变化、波动、期限利差。",
        change="在原始特征之外加入历史变化、滚动波动、期限结构等衍生变量；其他训练口径尽量保持和基准一致。",
        command="python scripts/run_curve_2020_rolling_validation.py --out-dir models/experiments/curve_2020_rolling_derived_features --derive-features --window 60 --horizon 5 --models gru,tcn,transformer --device cuda",
        json_path=ROOT / "data/backtests/experiments/derived_features_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/derived_features_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "models/experiments/curve_2020_rolling_derived_features/rolling_validation_summary.csv",
        conclusion="收益略高于基准，但分类指标下降，说明部分派生特征可能增加了噪音；它有价值，但需要筛选。",
        action="暂不直接上日报，先做特征筛选和共线性压缩。",
        caveat="因为部分派生特征需要20日历史，样本区间比基准短，不能只看累计收益做绝对比较。",
    ),
    Experiment(
        key="window120",
        short_name="120日窗口",
        full_name="方法二A：120日长窗口",
        hypothesis="信用债趋势和利率周期变化可能比60个交易日更慢，模型需要看更长历史才能识别趋势背景。",
        change="只把输入窗口从60日拉长到120日；标签、模型类型、回测规则保持不变。",
        command="python scripts/run_curve_2020_rolling_validation.py --out-dir models/experiments/curve_2020_rolling_window120 --window 120 --horizon 5 --models gru,tcn,transformer --device cuda",
        json_path=ROOT / "data/backtests/experiments/window120_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/window120_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "models/experiments/curve_2020_rolling_window120/rolling_validation_summary.csv",
        conclusion="累计收益只小幅提高，但最大回撤明显下降，说明长窗口对风险控制有帮助。",
        action="不单独替代基准，适合作为双窗口集成的一部分。",
        caveat="长窗口反应更慢，可能错过急转行情。",
    ),
    Experiment(
        key="window60_120",
        short_name="双窗口集成",
        full_name="方法二：60日 + 120日双窗口集成",
        hypothesis="短窗口更敏感，长窗口更稳健。把两类模型概率平均，可能比单一窗口更稳定。",
        change="合并60日窗口基准和120日窗口模型；同一期限、同一日期上对看多/看空/震荡概率取平均，再进入仓位规则。",
        command="合并 models/curve_2020_rolling_validation 与 models/experiments/curve_2020_rolling_window120 的滚动预测，再用 scripts/run_simple_portfolio_backtest.py 回测。",
        json_path=ROOT / "data/backtests/experiments/window60_120_ensemble_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/window60_120_ensemble_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "data/backtests/experiments/merged_summaries/window60_120_merged_summary.csv",
        conclusion="这是最适合立刻接入日报的提升：仍然是5日信号，但累计收益和年化都明显超过基准。",
        action="建议作为日报5日信号的主版本上线，同时保留旧基准并行观察。",
        caveat="本质是集成，不是全新预测逻辑；未来实盘仍需并行监控。",
    ),
    Experiment(
        key="h10",
        short_name="10日标签",
        full_name="方法三：10日标签",
        hypothesis="5日方向可能太噪，信用债交易更偏中期，预测未来10个交易日也许更贴近实际决策节奏。",
        change="把标签从未来5个交易日改为未来10个交易日；模型类型、滚动训练和仓位回测规则基本保持一致。",
        command="python scripts/run_curve_2020_rolling_validation.py --out-dir models/experiments/curve_2020_rolling_h10_all_models --horizon 10 --window 60 --models gru,tcn,transformer --device cuda",
        json_path=ROOT / "data/backtests/experiments/h10_all_models_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/h10_all_models_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "models/experiments/curve_2020_rolling_h10_all_models/rolling_validation_summary.csv",
        conclusion="本轮收益最高，说明把预测周期拉长能显著降低短期噪声，更符合信用债趋势资产的特征。",
        action="建议新增为日报里的中期趋势模块，而不是直接和5日信号混成一个结论。",
        caveat="它预测的是未来10日方向，语义已经不同，不能简单说它就是5日模型的升级版。",
    ),
    Experiment(
        key="big",
        short_name="大模型",
        full_name="方法四：加大模型容量",
        hypothesis="RTX 4090 能承载更大的网络，测试单纯加大模型是否能提升结果。",
        change="hidden_size 从64提高到128，层数从2提高到3，Transformer heads 从4提高到8，dropout 提高到0.25，训练轮数提高到16。",
        command="python scripts/run_curve_2020_rolling_validation.py --out-dir models/experiments/curve_2020_rolling_big_h5 --hidden-size 128 --layers 3 --heads 8 --dropout 0.25 --lr 0.0007 --epochs 16 --patience 4 --window 60 --horizon 5 --device cuda",
        json_path=ROOT / "data/backtests/experiments/big_h5_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/big_h5_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "models/experiments/curve_2020_rolling_big_h5/rolling_validation_summary.csv",
        conclusion="这次明确试了更大模型，结果低于基准。当前瓶颈更像是样本长度、标签噪声和特征质量，而不是4090算力没用满。",
        action="暂不进入主线，但保留为后续更长历史数据下的候选。",
        caveat="不是永远不能用大模型；只是2020年以来这批数据不够支撑单纯加容量。",
    ),
    Experiment(
        key="combo",
        short_name="三路组合",
        full_name="额外验证：60日 + 120日 + 10日标签三路组合",
        hypothesis="如果模型越多越好，那么把短窗口、长窗口、中期标签都加进去应该继续提升。",
        change="把60日窗口、120日窗口、10日标签三组滚动预测合并，概率平均后回测。",
        command="合并 baseline、window120、h10 三套 rolling summary，再用 scripts/run_simple_portfolio_backtest.py 回测。",
        json_path=ROOT / "data/backtests/experiments/window60_120_h10_ensemble_portfolio/portfolio_backtest_report.json",
        portfolio_summary_path=ROOT / "data/backtests/experiments/window60_120_h10_ensemble_portfolio/portfolio_summary.csv",
        validation_summary_path=ROOT / "data/backtests/experiments/merged_summaries/window60_120_h10_merged_summary.csv",
        conclusion="没有超过单独10日标签，说明模型不是越多越好；不同预测周期硬混合会稀释有效信号。",
        action="不作为主线，只作为反证保留。",
        caveat="组合结果仍优于旧基准，但其解释性不如双窗口和10日标签清楚。",
    ),
]


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MSYH", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", str(FONT_BOLD)))
    if PLOT_FONT.exists():
        font_manager.fontManager.addfont(str(PLOT_FONT))
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def pp(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value) * 100:.{digits}f}个百分点"


def cn_label(text: str) -> str:
    return (
        str(text)
        .replace("AAA 3年", "3年")
        .replace("AAA 5年", "5年")
        .replace("AAA 10年", "10年")
        .replace("AAA+ 20年", "20年")
    )


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def path_p(text: object, style: ParagraphStyle) -> Paragraph:
    value = escape(str(text))
    value = value.replace("\\", " / ").replace("/", "/ ")
    return Paragraph(value, style)


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
            textColor=colors.HexColor("#15202b"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=9.3,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#617083"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="MSYH-Bold",
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#15202b"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="MSYH-Bold",
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#15202b"),
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=8.8,
            leading=13.5,
            textColor=colors.HexColor("#25313d"),
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=7.2,
            leading=10.2,
            textColor=colors.HexColor("#617083"),
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=7.1,
            leading=9.2,
            textColor=colors.HexColor("#25313d"),
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            parent=base["BodyText"],
            fontName="MSYH-Bold",
            fontSize=7.2,
            leading=9.4,
            textColor=colors.HexColor("#15202b"),
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="MSYH-Bold",
            fontSize=9.1,
            leading=14.5,
            textColor=colors.HexColor("#17324d"),
            backColor=colors.HexColor("#eef5ff"),
            borderColor=colors.HexColor("#2f6f9f"),
            borderWidth=0.8,
            borderPadding=7,
            spaceAfter=7,
        ),
        "warning": ParagraphStyle(
            "warning",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#4c3715"),
            backColor=colors.HexColor("#fff7e8"),
            borderColor=colors.HexColor("#bd8429"),
            borderWidth=0.8,
            borderPadding=7,
            spaceAfter=7,
        ),
    }


def table_style(header="#eef2f7", first_col_bold=False) -> TableStyle:
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "MSYH-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "MSYH"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 9.4),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15202b")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e0e8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if first_col_bold:
        commands.append(("FONTNAME", (0, 1), (0, -1), "MSYH-Bold"))
    return TableStyle(commands)


def load_report(exp: Experiment) -> dict:
    return json.loads(exp.json_path.read_text(encoding="utf-8"))


def load_metrics() -> pd.DataFrame:
    baseline_report = load_report(EXPERIMENTS[0])["portfolio_metrics"]
    rows = []
    for exp in EXPERIMENTS:
        report = load_report(exp)
        m = report["portfolio_metrics"]
        rows.append(
            {
                "key": exp.key,
                "short_name": exp.short_name,
                "full_name": exp.full_name,
                "start_date": m["start_date"],
                "end_date": m["end_date"],
                "periods": m["periods"],
                "total_return": m["total_return"],
                "annualized_return": m["annualized_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m.get("sharpe"),
                "positive_day_ratio": m.get("positive_day_ratio"),
                "average_position": m.get("average_position"),
                "turnover": m.get("turnover"),
                "buy_hold_total_return": m.get("buy_hold_total_return"),
                "neutral_total_return": m.get("neutral_total_return"),
                "excess_vs_baseline": m["total_return"] - baseline_report["total_return"],
            }
        )
    return pd.DataFrame(rows)


def load_validation(exp: Experiment) -> pd.DataFrame:
    if exp.validation_summary_path is None or not exp.validation_summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(exp.validation_summary_path, encoding="utf-8-sig")


def validation_overview(exp: Experiment) -> dict[str, object]:
    df = load_validation(exp)
    if df.empty:
        return {
            "rows": "-",
            "folds": "-",
            "models": "-",
            "accuracy": None,
            "macro_f1": None,
            "active_ratio": None,
            "active_win": None,
        }
    return {
        "rows": len(df),
        "folds": df["fold"].nunique() if "fold" in df else "-",
        "models": "、".join(sorted(map(str, df["model"].dropna().unique()))) if "model" in df else "-",
        "accuracy": pd.to_numeric(df.get("accuracy"), errors="coerce").mean(),
        "macro_f1": pd.to_numeric(df.get("macro_f1"), errors="coerce").mean(),
        "active_ratio": pd.to_numeric(df.get("proxy_active_signal_ratio"), errors="coerce").mean()
        if "proxy_active_signal_ratio" in df
        else None,
        "active_win": pd.to_numeric(df.get("proxy_positive_active_ratio"), errors="coerce").mean()
        if "proxy_positive_active_ratio" in df
        else None,
    }


def load_portfolio_summary(exp: Experiment) -> pd.DataFrame:
    df = pd.read_csv(exp.portfolio_summary_path, encoding="utf-8-sig")
    if "signal_counts" in df:
        df["signal_counts_dict"] = df["signal_counts"].apply(lambda x: ast.literal_eval(str(x)) if str(x).strip() else {})
    else:
        df["signal_counts_dict"] = [{} for _ in range(len(df))]
    return df


def make_return_bar(metrics: pd.DataFrame, path: Path) -> None:
    labels = metrics["short_name"].tolist()
    values = metrics["total_return"].to_numpy() * 100
    colors_bar = ["#8a97a6", "#2f6f9f", "#5b7c99", "#238a62", "#1f78b4", "#c74343", "#b7812f"]
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    bars = ax.bar(labels, values, color=colors_bar[: len(labels)], width=0.62)
    base = float(metrics.loc[metrics["key"].eq("baseline"), "total_return"].iloc[0]) * 100
    ax.axhline(base, color="#3c4650", linewidth=1.1, linestyle="--", alpha=0.7, label="旧基准")
    ax.set_ylabel("累计收益率（%）")
    ax.set_title("各实验累计收益率对比")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.28, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_risk_return_chart(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    colors_map = {
        "baseline": "#8a97a6",
        "derived": "#2f6f9f",
        "window120": "#5b7c99",
        "window60_120": "#238a62",
        "h10": "#1f78b4",
        "big": "#c74343",
        "combo": "#b7812f",
    }
    for _, row in metrics.iterrows():
        x = abs(row["max_drawdown"]) * 100
        y = row["annualized_return"] * 100
        size = 70 + row["average_position"] * 120
        ax.scatter(x, y, s=size, color=colors_map[row["key"]], alpha=0.82)
        ax.text(x + 0.03, y + 0.02, row["short_name"], fontsize=9)
    ax.set_xlabel("最大回撤绝对值（%），越靠左越好")
    ax.set_ylabel("年化收益率（%），越靠上越好")
    ax.set_title("收益 - 回撤位置图")
    ax.grid(alpha=0.23)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_net_value_chart(path: Path) -> None:
    items = [
        ("旧基准", ROOT / "data/backtests/curve_2020_rolling_portfolio/portfolio_detail.csv", "#8a97a6"),
        ("双窗口集成", ROOT / "data/backtests/experiments/window60_120_ensemble_portfolio/portfolio_detail.csv", "#238a62"),
        ("10日标签", ROOT / "data/backtests/experiments/h10_all_models_portfolio/portfolio_detail.csv", "#1f78b4"),
        ("大模型", ROOT / "data/backtests/experiments/big_h5_portfolio/portfolio_detail.csv", "#c74343"),
        ("三路组合", ROOT / "data/backtests/experiments/window60_120_h10_ensemble_portfolio/portfolio_detail.csv", "#b7812f"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for label, csv_path, color in items:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        ax.plot(pd.to_datetime(df["date"]), df["strategy_value"], label=label, color=color, linewidth=2.0)
    ax.set_title("主要实验净值曲线")
    ax.set_ylabel("组合净值")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_tenor_heatmap(exps: list[Experiment], path: Path) -> None:
    rows = []
    for exp in exps:
        df = load_portfolio_summary(exp)
        for _, row in df.iterrows():
            rows.append(
                {
                    "experiment": exp.short_name,
                    "tenor": cn_label(row["tenor"]),
                    "return": row["total_return"] * 100,
                }
            )
    data = pd.DataFrame(rows).pivot(index="experiment", columns="tenor", values="return")
    data = data[[col for col in ["3年", "5年", "10年", "20年"] if col in data.columns]]
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    im = ax.imshow(data.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set_title("各期限累计收益率（%）")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="#16202b", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_signal_stack(exp: Experiment, path: Path) -> None:
    df = load_portfolio_summary(exp)
    rows = []
    for _, row in df.iterrows():
        counts = row["signal_counts_dict"]
        total = sum(counts.values()) or 1
        rows.append(
            {
                "tenor": cn_label(row["tenor"]),
                "看空": counts.get("看空", 0) / total,
                "看多": counts.get("看多", 0) / total,
                "震荡": counts.get("震荡", 0) / total,
            }
        )
    plot = pd.DataFrame(rows).set_index("tenor")
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    bottom = pd.Series(0.0, index=plot.index)
    colors_map = {"看空": "#c74343", "看多": "#238a62", "震荡": "#bd8429"}
    for col in ["看空", "看多", "震荡"]:
        ax.bar(plot.index, plot[col] * 100, bottom=bottom * 100, label=col, color=colors_map[col])
        bottom += plot[col]
    ax.set_ylabel("信号占比（%）")
    ax.set_title(f"{exp.short_name}：各期限信号结构")
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metrics_table(metrics: pd.DataFrame, s: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [
            p("实验", s["table_bold"]),
            p("区间", s["table_bold"]),
            p("累计", s["table_bold"]),
            p("年化", s["table_bold"]),
            p("最大回撤", s["table_bold"]),
            p("夏普", s["table_bold"]),
            p("平均仓位", s["table_bold"]),
            p("换手", s["table_bold"]),
            p("相对基准", s["table_bold"]),
        ]
    ]
    for _, row in metrics.iterrows():
        rows.append(
            [
                p(row["short_name"], s["table"]),
                p(f"{row['start_date']} 至 {row['end_date']}", s["table"]),
                p(pct(row["total_return"]), s["table"]),
                p(pct(row["annualized_return"]), s["table"]),
                p(pct(row["max_drawdown"]), s["table"]),
                p(num(row["sharpe"]), s["table"]),
                p(pct(row["average_position"], 1), s["table"]),
                p(num(row["turnover"], 2), s["table"]),
                p(pp(row["excess_vs_baseline"]), s["table"]),
            ]
        )
    table = Table(rows, colWidths=[2.2 * cm, 3.3 * cm, 1.45 * cm, 1.45 * cm, 1.55 * cm, 1.15 * cm, 1.45 * cm, 1.2 * cm, 2.05 * cm])
    table.setStyle(table_style(first_col_bold=True))
    return table


def validation_table(exps: list[Experiment], s: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [
            p("实验", s["table_bold"]),
            p("训练记录", s["table_bold"]),
            p("折数", s["table_bold"]),
            p("模型类型", s["table_bold"]),
            p("平均准确率", s["table_bold"]),
            p("Macro F1", s["table_bold"]),
            p("主动信号占比", s["table_bold"]),
            p("主动信号胜率", s["table_bold"]),
        ]
    ]
    for exp in exps:
        v = validation_overview(exp)
        rows.append(
            [
                p(exp.short_name, s["table"]),
                p(v["rows"], s["table"]),
                p(v["folds"], s["table"]),
                p(v["models"], s["table"]),
                p(pct(v["accuracy"]), s["table"]),
                p(pct(v["macro_f1"]), s["table"]),
                p(pct(v["active_ratio"]), s["table"]),
                p(pct(v["active_win"]), s["table"]),
            ]
        )
    table = Table(rows, colWidths=[2.1 * cm, 1.45 * cm, 1.0 * cm, 3.2 * cm, 1.7 * cm, 1.55 * cm, 1.9 * cm, 1.9 * cm])
    table.setStyle(table_style("#edf6ef", first_col_bold=True))
    return table


def tenor_table(exp: Experiment, s: dict[str, ParagraphStyle]) -> Table:
    df = load_portfolio_summary(exp)
    rows = [
        [
            p("期限", s["table_bold"]),
            p("累计", s["table_bold"]),
            p("年化", s["table_bold"]),
            p("最大回撤", s["table_bold"]),
            p("平均仓位", s["table_bold"]),
            p("换手", s["table_bold"]),
            p("信号分布", s["table_bold"]),
        ]
    ]
    for _, row in df.iterrows():
        counts = row["signal_counts_dict"]
        signal_text = " / ".join([f"{k}{v}" for k, v in counts.items()])
        rows.append(
            [
                p(cn_label(row["tenor"]), s["table"]),
                p(pct(row["total_return"]), s["table"]),
                p(pct(row["annualized_return"]), s["table"]),
                p(pct(row["max_drawdown"]), s["table"]),
                p(pct(row["average_position"], 1), s["table"]),
                p(num(row["turnover"], 2), s["table"]),
                p(signal_text, s["table"]),
            ]
        )
    table = Table(rows, colWidths=[1.55 * cm, 1.35 * cm, 1.35 * cm, 1.45 * cm, 1.45 * cm, 1.15 * cm, 6.0 * cm])
    table.setStyle(table_style("#f7f3e8", first_col_bold=True))
    return table


def exp_detail_table(exp: Experiment, s: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [p("实验假设", s["table_bold"]), p(exp.hypothesis, s["table"])],
        [p("实际改动", s["table_bold"]), p(exp.change, s["table"])],
        [p("运行命令", s["table_bold"]), path_p(exp.command, s["table"])],
        [p("结果判断", s["table_bold"]), p(exp.conclusion, s["table"])],
        [p("处理建议", s["table_bold"]), p(exp.action, s["table"])],
        [p("注意事项", s["table_bold"]), p(exp.caveat, s["table"])],
    ]
    table = Table(rows, colWidths=[2.1 * cm, 14.4 * cm])
    table.setStyle(table_style("#eef5ff", first_col_bold=True))
    return table


def files_table(exps: list[Experiment], s: dict[str, ParagraphStyle]) -> Table:
    rows = [[p("实验", s["table_bold"]), p("回测报告", s["table_bold"]), p("滚动训练/合并汇总", s["table_bold"])]]
    for exp in exps:
        rows.append(
            [
                p(exp.short_name, s["table"]),
                path_p(str(exp.json_path.relative_to(ROOT)).replace(".json", ".html"), s["small"]),
                path_p(str(exp.validation_summary_path.relative_to(ROOT)) if exp.validation_summary_path else "-", s["small"]),
            ]
        )
    table = Table(rows, colWidths=[2.4 * cm, 7.0 * cm, 7.1 * cm])
    table.setStyle(table_style(first_col_bold=True))
    return table


def build_pdf() -> Path:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    s = styles()
    metrics = load_metrics()

    return_bar = TMP_DIR / "return_bar.png"
    risk_return = TMP_DIR / "risk_return.png"
    net_value = TMP_DIR / "net_value.png"
    tenor_heatmap = TMP_DIR / "tenor_heatmap.png"
    h10_signal = TMP_DIR / "h10_signal.png"
    window_signal = TMP_DIR / "window_signal.png"
    make_return_bar(metrics, return_bar)
    make_risk_return_chart(metrics, risk_return)
    make_net_value_chart(net_value)
    make_tenor_heatmap(EXPERIMENTS, tenor_heatmap)
    make_signal_stack(next(exp for exp in EXPERIMENTS if exp.key == "h10"), h10_signal)
    make_signal_stack(next(exp for exp in EXPERIMENTS if exp.key == "window60_120"), window_signal)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.35 * cm,
        leftMargin=1.35 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.1 * cm,
    )
    story = []
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    baseline = metrics[metrics["key"].eq("baseline")].iloc[0]
    window = metrics[metrics["key"].eq("window60_120")].iloc[0]
    h10 = metrics[metrics["key"].eq("h10")].iloc[0]
    big = metrics[metrics["key"].eq("big")].iloc[0]

    story.append(Paragraph("信用债AI模型提升实验详版报告", s["title"]))
    story.append(
        Paragraph(
            f"生成时间：{generated}；数据：DM收益率曲线，2020年起；验证：滚动训练 / 滚动预测 / 不扣交易成本组合回测",
            s["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "一句话结论：这轮实验不是简单把模型变大，而是分别测试了特征、窗口、标签周期、模型容量四条路线。"
            f"结果显示，10日标签收益最高，累计 {pct(h10['total_return'])}；"
            f"最适合短期接入日报的是双窗口集成，累计 {pct(window['total_return'])}，相对旧基准提高 {pp(window['excess_vs_baseline'])}；"
            f"大模型累计 {pct(big['total_return'])}，低于旧基准 {pct(baseline['total_return'])}，说明当前瓶颈不是单纯算力。",
            s["callout"],
        )
    )

    story.append(Paragraph("一、实验问题到底是什么", s["h1"]))
    story.append(
        Paragraph(
            "用户提出的问题是：怎样提升信用债AI模型表现，不能只说“数据太短”或“不能把模型加大”。"
            "所以本轮实验固定一个核心目标：在不新增外部数据的前提下，找出现有数据还能榨出的改进空间。"
            "实验必须有对照组，并且每次只主要改变一个维度，避免把多个改动混在一起后无法解释。",
            s["body"],
        )
    )
    setup_rows = [
        [p("项目", s["table_bold"]), p("口径", s["table_bold"])],
        [p("预测对象", s["table"]), p("AAA 3年、AAA 5年、AAA 10年、AAA+ 20年信用债收益率方向", s["table"])],
        [p("模型类型", s["table"]), p("GRU、TCN、Transformer，均为神经网络时序模型；每个期限分别训练", s["table"])],
        [p("训练方式", s["table"]), p("滚动训练：用过去一段历史训练，在之后一段样本外预测，再向后滚动", s["table"])],
        [p("标签", s["table"]), p("看空=未来收益率上行，债券价格承压；看多=未来收益率下行，债券价格受益；震荡=变化不够大", s["table"])],
        [p("组合规则", s["table"]), p("哪个期限出现看多就给该期限加仓，出现看空就减仓，震荡不动；仓位限制在0到100%；本轮不扣交易成本", s["table"])],
        [p("基准意义", s["table"]), p("旧基准是60日窗口、5日标签、三模型集成；所有改进都和它比较", s["table"])],
    ]
    tbl = Table(setup_rows, colWidths=[3.2 * cm, 13.2 * cm])
    tbl.setStyle(table_style("#eef5ff", first_col_bold=True))
    story.append(tbl)

    story.append(Paragraph("二、总结果", s["h1"]))
    story.append(metrics_table(metrics, s))
    story.append(Paragraph("注意：派生特征与10日标签实验的可回测日期略有差异，表中已经列出区间；因此要同时看收益、年化、回撤和解释性。", s["warning"]))
    story.append(Spacer(1, 0.1 * cm))
    story.append(Image(str(return_bar), width=17.2 * cm, height=7.35 * cm))

    story.append(PageBreak())
    story.append(Paragraph("三、收益和风险图", s["h1"]))
    story.append(Image(str(net_value), width=17.2 * cm, height=7.95 * cm))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(risk_return), width=17.2 * cm, height=7.65 * cm))
    story.append(
        Paragraph(
            "读图方法：净值曲线看的是路径；收益-回撤图看的是同等回撤下谁的年化更高。"
            "双窗口集成的位置比较均衡，10日标签收益最高，大模型没有占优。",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("四、分类层面的结果", s["h1"]))
    story.append(validation_table(EXPERIMENTS, s))
    story.append(
        Paragraph(
            "这里要特别说明：分类准确率不是组合收益的唯一来源。信用债组合更关心关键波段上仓位是否站对，"
            "而不是每天三分类都猜对。因此10日标签的平均准确率不是最高，但它的仓位路径更赚钱。",
            s["warning"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(str(tenor_heatmap), width=16.4 * cm, height=7.9 * cm))
    story.append(
        Paragraph(
            "期限贡献上，20年期限因为久期更长，同方向行情下收益和回撤都会被放大；"
            "因此后续日报不能只看综合方向，也要继续保留各期限独立信号。",
            s["body"],
        )
    )

    for exp in EXPERIMENTS:
        story.append(PageBreak())
        story.append(Paragraph(f"五、单项实验：{exp.full_name}", s["h1"]))
        metric = metrics[metrics["key"].eq(exp.key)].iloc[0]
        summary_rows = [
            [p("累计收益", s["table_bold"]), p(pct(metric["total_return"]), s["table"]), p("年化收益", s["table_bold"]), p(pct(metric["annualized_return"]), s["table"])],
            [p("最大回撤", s["table_bold"]), p(pct(metric["max_drawdown"]), s["table"]), p("夏普", s["table_bold"]), p(num(metric["sharpe"]), s["table"])],
            [p("平均仓位", s["table_bold"]), p(pct(metric["average_position"], 1), s["table"]), p("相对基准", s["table_bold"]), p(pp(metric["excess_vs_baseline"]), s["table"])],
            [p("回测区间", s["table_bold"]), p(f"{metric['start_date']} 至 {metric['end_date']}", s["table"]), p("交易日数", s["table_bold"]), p(str(metric["periods"]), s["table"])],
        ]
        summary_tbl = Table(summary_rows, colWidths=[2.2 * cm, 3.9 * cm, 2.2 * cm, 3.9 * cm])
        summary_tbl.setStyle(table_style("#f7f9fc", first_col_bold=True))
        story.append(summary_tbl)
        story.append(Spacer(1, 0.15 * cm))
        story.append(exp_detail_table(exp, s))
        story.append(Paragraph("分期限回测", s["h2"]))
        story.append(tenor_table(exp, s))

        if exp.key == "window60_120":
            story.append(Spacer(1, 0.2 * cm))
            story.append(Image(str(window_signal), width=15.7 * cm, height=6.45 * cm))
        if exp.key == "h10":
            story.append(Spacer(1, 0.2 * cm))
            story.append(Image(str(h10_signal), width=15.7 * cm, height=6.45 * cm))

    story.append(PageBreak())
    story.append(Paragraph("六、这轮实验回答了什么", s["h1"]))
    answer_rows = [
        [p("问题", s["table_bold"]), p("实验回答", s["table_bold"])],
        [p("是不是喂更多特征就一定更准？", s["table"]), p("不是。派生特征有收益增量，但分类指标下降，说明特征需要筛选，不能盲目堆。", s["table"])],
        [p("是不是看更长历史有用？", s["table"]), p("有用，尤其对回撤控制有帮助。120日窗口单独收益只小幅改善，但和60日窗口集成后提升明显。", s["table"])],
        [p("是不是标签周期要改？", s["table"]), p("很可能要。10日标签是本轮收益最高的方法，说明信用债更适合中期趋势视角。", s["table"])],
        [p("是不是4090就应该跑大模型？", s["table"]), p("已经试了更大网络，本轮不如基准。算力有价值，但当前更缺的是更长历史、更好标签和更干净特征。", s["table"])],
        [p("为什么不能只看准确率？", s["table"]), p("因为组合收益来自仓位路径。模型可以不是每天都猜对，但只要关键趋势段加对仓、风险段少犯错，就可能提升收益。", s["table"])],
    ]
    tbl = Table(answer_rows, colWidths=[5.0 * cm, 11.5 * cm])
    tbl.setStyle(table_style("#eef5ff", first_col_bold=True))
    story.append(tbl)

    story.append(Paragraph("七、建议路线", s["h1"]))
    rec_rows = [
        [p("优先级", s["table_bold"]), p("动作", s["table_bold"]), p("理由", s["table_bold"])],
        [p("第一优先级", s["table"]), p("把双窗口集成接入日报5日信号", s["table"]), p("同为5日标签，语义不变，收益和年化均超过基准，解释性也清楚。", s["table"])],
        [p("第二优先级", s["table"]), p("新增10日中期趋势信号", s["table"]), p("收益最高，但标签语义不同，应该独立展示，不能混成一句综合结论。", s["table"])],
        [p("第三优先级", s["table"]), p("派生特征做筛选后再复测", s["table"]), p("已有轻微增量，但需要避免噪音和过拟合。", s["table"])],
        [p("暂缓", s["table"]), p("大模型不进入主线", s["table"]), p("本轮证据不支持单纯加容量；等数据更长或有预训练/正则方案再试。", s["table"])],
    ]
    tbl = Table(rec_rows, colWidths=[2.6 * cm, 5.0 * cm, 8.9 * cm])
    tbl.setStyle(table_style("#edf6ef", first_col_bold=True))
    story.append(tbl)
    story.append(Paragraph("八、可复现文件", s["h1"]))
    story.append(files_table(EXPERIMENTS, s))

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("MSYH", 7)
        canvas.setFillColor(colors.HexColor("#617083"))
        canvas.drawRightString(A4[0] - 1.35 * cm, 0.62 * cm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
