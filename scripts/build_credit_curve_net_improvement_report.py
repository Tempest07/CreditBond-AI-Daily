from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "model_arena" / "curve_2020_credit_curve_net_improvement_v1"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number * 100:.{digits}f}%"


def _pp(value: float, digits: int = 1) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{digits}f} pp"


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, encoding="utf-8-sig")


def _row(df: pd.DataFrame, tenor: str) -> pd.Series:
    found = df[df["tenor"] == tenor]
    if found.empty:
        raise ValueError(f"missing tenor: {tenor}")
    return found.iloc[0]


def _metric_row(tenor: str, base: pd.Series, target: pd.Series) -> str:
    f1_delta = float(target["cal_guard_eval_macro_f1"]) - float(base["cal_guard_eval_macro_f1"])
    ece_delta = float(target["cal_eval_ece"]) - float(base["cal_eval_ece"])
    ret_delta = float(target["cal_guard_eval_total_return_proxy"]) - float(base["cal_guard_eval_total_return_proxy"])
    raw_f1_delta = float(target["raw_eval_macro_f1"]) - float(base["raw_eval_macro_f1"])
    raw_ret_delta = float(target["raw_eval_total_return_proxy"]) - float(base["raw_eval_total_return_proxy"])
    verdict = "采用" if tenor == "20年" else "观察"
    if tenor in {"3年", "5年"}:
        verdict = "暂不采用"
    if tenor == "10年":
        verdict = "只作研究线索"
    return f"""
      <tr>
        <td>{tenor}</td>
        <td>{_pct(base['cal_guard_eval_macro_f1'])} -> {_pct(target['cal_guard_eval_macro_f1'])}<small>{_pp(f1_delta)}</small></td>
        <td>{_pct(base['cal_eval_ece'])} -> {_pct(target['cal_eval_ece'])}<small>{_pp(ece_delta)}</small></td>
        <td>{_pct(base['cal_guard_eval_total_return_proxy'])} -> {_pct(target['cal_guard_eval_total_return_proxy'])}<small>{_pp(ret_delta)}</small></td>
        <td>{_pct(base['raw_eval_macro_f1'])} -> {_pct(target['raw_eval_macro_f1'])}<small>{_pp(raw_f1_delta)}</small></td>
        <td>{_pct(base['raw_eval_total_return_proxy'])} -> {_pct(target['raw_eval_total_return_proxy'])}<small>{_pp(raw_ret_delta)}</small></td>
        <td><span class="tag">{verdict}</span></td>
      </tr>
    """


def _selected_row(row: pd.Series) -> str:
    return f"""
      <tr>
        <td>{row['tenor']}</td>
        <td>Top{int(row['member_count'])}/{int(row['total_available_models'])}</td>
        <td>{row['chosen_signal_layer_cn']}</td>
        <td>{_pct(row['selected_eval_macro_f1'])}</td>
        <td>{_pct(row['selected_eval_ece'])}</td>
        <td>{_pct(row['selected_eval_total_return_proxy'])}</td>
        <td>{_pct(row['selected_eval_active_ratio'])}</td>
      </tr>
    """


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _read("data/model_arena/curve_2020_credit_curve_net_ensemble_v1/credit_curve_net_ensemble_summary.csv")
    target = _read("data/model_arena/curve_2020_credit_curve_net_target_feature_v1/credit_curve_net_ensemble_summary.csv")
    selected = _read("data/model_arena/curve_2020_credit_curve_net_selected_ensemble_v1/selected_ensemble_summary.csv")
    base_meta = json.loads((ROOT / "data/model_arena/curve_2020_credit_curve_net_ensemble_v1/credit_curve_net_ensemble_meta.json").read_text(encoding="utf-8"))
    target_meta = json.loads((ROOT / "data/model_arena/curve_2020_credit_curve_net_target_feature_v1/credit_curve_net_ensemble_meta.json").read_text(encoding="utf-8"))

    tenors = ["3年", "5年", "10年", "20年"]
    comparison_rows = "\n".join(_metric_row(tenor, _row(base, tenor), _row(target, tenor)) for tenor in tenors)
    selected_rows = "\n".join(_selected_row(row) for _, row in selected.sort_values("tenor").iterrows())

    base_avg_f1 = float(base["cal_guard_eval_macro_f1"].mean())
    target_avg_f1 = float(target["cal_guard_eval_macro_f1"].mean())
    base_avg_ece = float(base["cal_eval_ece"].mean())
    target_avg_ece = float(target["cal_eval_ece"].mean())
    base_avg_ret = float(base["cal_guard_eval_total_return_proxy"].mean())
    target_avg_ret = float(target["cal_guard_eval_total_return_proxy"].mean())

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CreditCurveNet 自研模型提升实验报告</title>
  <style>
    :root {{ --ink:#17202c; --muted:#627086; --line:#dbe4ee; --paper:#fff; --bg:#eef3f8; --red:#d94a45; --green:#21815b; --blue:#315f9f; --gold:#b9812a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    .wrap {{ width:min(1180px, calc(100% - 36px)); margin:0 auto; }}
    .hero {{ padding:34px 0 24px; background:linear-gradient(180deg,#fbfcff,#edf3f8); border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--blue); font-weight:800; }}
    h1 {{ margin:8px 0 10px; font-size:38px; line-height:1.12; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:22px; }}
    h3 {{ margin:0 0 8px; font-size:17px; }}
    p {{ line-height:1.75; }}
    .muted,.note {{ color:var(--muted); }}
    .main {{ padding:20px 0 42px; display:grid; gap:16px; }}
    .section,.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 10px 24px rgba(35,48,66,.05); }}
    .section {{ padding:20px; }}
    .takeaway {{ margin:0; font-size:17px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:16px; }}
    .metric {{ background:#f6f8fb; border-top:3px solid var(--blue); border-radius:6px; padding:10px 12px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric b {{ display:block; margin-top:5px; font-size:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ padding:16px; }}
    .card.good {{ border-top:4px solid var(--red); }}
    .card.warn {{ border-top:4px solid var(--gold); }}
    .card.bad {{ border-top:4px solid var(--green); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:980px; background:#fff; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid #edf1f5; text-align:left; font-size:13px; white-space:nowrap; vertical-align:top; }}
    th {{ color:#526173; background:#f7f9fb; font-weight:700; }}
    td small {{ display:block; color:var(--muted); margin-top:3px; }}
    .tag {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#f7e6df; color:#b3332d; font-weight:800; }}
    .rules {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .rules div {{ background:#f6f8fb; border-left:4px solid var(--gold); border-radius:6px; padding:10px 12px; }}
    @media (max-width:900px) {{ .cards,.rules,.metric-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:30px; }} }}
    @media print {{ @page {{ size:A4; margin:8mm; }} body {{ background:#fff; }} .wrap {{ width:100%; }} h1 {{ font-size:26px; }} .section,.card {{ box-shadow:none; break-inside:avoid; }} .cards,.rules {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} th,td {{ font-size:10px; padding:6px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 研究辅助 - 自研神经网络提升实验</div>
      <h1>这次真正有效的是：分期限纳入自身收益率历史，而不是盲目堆大模型</h1>
      <p class="muted">本轮先跑满 20.8 分钟训练 155 个抗过拟合多种子模型；随后做嵌套筛选验证；最后追加 8.2 分钟训练 60 个“纳入自身历史特征”的消融模型。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="takeaway">20年期限最值得升级：纳入自身收益率历史后，保护后宏 F1 从 {_pct(_row(base, "20年")["cal_guard_eval_macro_f1"])} 提升到 {_pct(_row(target, "20年")["cal_guard_eval_macro_f1"])}，代理收益从 {_pct(_row(base, "20年")["cal_guard_eval_total_return_proxy"])} 改善到 {_pct(_row(target, "20年")["cal_guard_eval_total_return_proxy"])}。3年和5年暂不切换；10年只作为研究信号，因为原始方向变强，但保护后没有稳定胜出。</p>
      <div class="metric-grid">
        <div class="metric"><span>20分钟长跑</span><b>{int(base_meta['trained_models'])} 个模型</b></div>
        <div class="metric"><span>追加消融</span><b>{int(target_meta['trained_models'])} 个模型</b></div>
        <div class="metric"><span>平均保护后宏 F1</span><b>{_pct(base_avg_f1)} -> {_pct(target_avg_f1)}</b></div>
        <div class="metric"><span>平均保护后收益</span><b>{_pct(base_avg_ret)} -> {_pct(target_avg_ret)}</b></div>
      </div>
    </section>
    <section class="cards">
      <article class="card good"><h3>20年：采用</h3><p>这是最明确的提升。F1、ECE、代理收益同时改善，且不是靠把信号全部压成震荡。</p></article>
      <article class="card warn"><h3>10年：研究线索</h3><p>原始方向明显更强，原始收益也更高，但经过保护层后优势被削弱，暂不直接替换主信号。</p></article>
      <article class="card bad"><h3>5年：暂不采用</h3><p>纳入自身历史后保护后 F1 和 ECE 都变差，说明该期限更容易被自身短期走势噪声干扰。</p></article>
      <article class="card bad"><h3>3年：暂不采用</h3><p>保护后 F1 基本不变，收益没有改善。短端仍需要更多资金面和流动性特征，而不是只改网络结构。</p></article>
    </section>
    <section class="section">
      <h2>核心对比</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>保护后宏 F1</th><th>校准 ECE</th><th>保护后代理收益</th><th>原始宏 F1</th><th>原始代理收益</th><th>建议</th></tr></thead>
          <tbody>{comparison_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>嵌套筛选实验告诉我们的事</h2>
      <p class="note">我额外做了“先筛子模型、再选 TopN、最后验收”的嵌套实验。它没有被选为主升级路径，因为 10年和20年容易在选择期看起来很好、样本外回落。这是防过拟合流程发挥作用的例子。</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>入选子模型</th><th>最终层</th><th>最终宏 F1</th><th>最终 ECE</th><th>最终收益</th><th>活跃信号</th></tr></thead>
          <tbody>{selected_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>防过拟合约束</h2>
      <div class="rules">
        <div><b>验收期不参与调参</b><br><span class="muted">训练、筛选、校准和验收按时间顺序拆开。</span></div>
        <div><b>保留失败实验</b><br><span class="muted">5年、3年没有硬说提升，避免漂亮但危险的结论。</span></div>
        <div><b>分期限采用</b><br><span class="muted">不因为20年有效就把所有期限一起替换。</span></div>
        <div><b>下一步滚动验证</b><br><span class="muted">20年升级前仍应跑滚动训练，观察不同市场阶段稳定性。</span></div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    html_path = OUT_DIR / "credit_curve_net_improvement_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
