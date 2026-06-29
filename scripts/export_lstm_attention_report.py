from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_ORIGINAL_REALPATH = os.path.realpath


def _safe_realpath(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> str:
    try:
        return _ORIGINAL_REALPATH(path, *args, **kwargs)
    except OSError:
        return os.path.abspath(path)


os.path.realpath = _safe_realpath

import torch

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.data import load_scaler, load_wide_dataset
from creditbond_ai.predict import load_checkpoint
from creditbond_ai.training import choose_device


LABEL_CN = {
    0: "看空",
    1: "看多",
    2: "震荡",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latest LSTM attention explanation report.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--rolling-summary", default="models/curve_2020_rolling_lstm_attention/rolling_validation_summary.csv")
    parser.add_argument("--model-dir", action="append", default=[])
    parser.add_argument("--out-dir", default="data/model_explain/lstm_attention_latest")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def safe_name(text: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(text), flags=re.UNICODE)
    return text.strip("_") or "item"


def infer_tenor(target_col: str) -> str:
    match = re.search(r":(\d+)\s*年", str(target_col))
    return match.group(1) if match else ""


def tenor_sort_key(target_col: str) -> float:
    tenor = infer_tenor(target_col)
    return float(tenor) if tenor else 999.0


def tenor_label(target_col: str) -> str:
    tenor = infer_tenor(target_col)
    rating = "AAA+" if "AAA+" in str(target_col) else "AAA"
    return f"{rating} {tenor}年" if tenor else str(target_col)


def latest_attention_models(summary_path: str | Path) -> list[Path]:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到滚动训练汇总表：{summary_path}")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    required = {"fold", "target_col", "model", "model_dir"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"滚动训练汇总表缺少字段：{sorted(missing)}")

    summary = summary[summary["model"].astype(str).str.lower().eq("lstm_attention")].copy()
    if summary.empty:
        raise ValueError(f"汇总表里没有 lstm_attention 模型：{summary_path}")

    summary["fold"] = pd.to_numeric(summary["fold"], errors="coerce")
    summary = summary.dropna(subset=["fold"])
    rows = []
    for _, group in summary.sort_values("fold").groupby("target_col", sort=False):
        rows.append(group.iloc[-1])
    rows.sort(key=lambda row: tenor_sort_key(str(row["target_col"])))
    return [Path(str(row["model_dir"])) for row in rows]


def load_features(path: str | Path) -> pd.DataFrame:
    return (
        load_wide_dataset(path)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .dropna(how="any")
        .reset_index(drop=True)
    )


def explain_model(model_dir: Path, features: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(model_dir, device)
    if not hasattr(model, "forward_with_attention"):
        raise ValueError(f"模型不支持注意力解释：{model_dir}")

    scaler = load_scaler(model_dir / "scaler.joblib")
    feature_cols = list(checkpoint["feature_cols"])
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise ValueError(f"{model_dir} 所需特征缺失：{missing[:8]}")

    window = int(checkpoint["window"])
    if len(features) < window:
        raise ValueError(f"数据长度不足，至少需要 {window} 行，当前 {len(features)} 行。")

    X = features[feature_cols].astype(float).to_numpy()
    X_scaled = scaler.transform(X).astype(np.float32)
    latest = torch.from_numpy(X_scaled[-window:][None, :, :]).float().to(device)

    with torch.no_grad():
        logits, weights = model.forward_with_attention(latest)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        attention = weights.detach().cpu().numpy()[0]

    window_df = features.iloc[-window:].copy().reset_index(drop=True)
    window_df["date"] = pd.to_datetime(window_df["date"])
    target_col = str(checkpoint.get("target_col", ""))
    detail = pd.DataFrame(
        {
            "date": window_df["date"].dt.strftime("%Y-%m-%d"),
            "days_before_latest": list(range(window - 1, -1, -1)),
            "attention_weight": attention,
        }
    )
    if target_col in window_df.columns:
        detail["target_yield"] = window_df[target_col].astype(float).to_numpy()
        detail["target_yield_change_1d"] = window_df[target_col].astype(float).diff(1).to_numpy()

    detail = detail.sort_values("attention_weight", ascending=False).reset_index(drop=True)
    detail["attention_rank"] = np.arange(1, len(detail) + 1)
    detail = detail.sort_values("date").reset_index(drop=True)

    pred_label = int(np.argmax(probs))
    latest_date = pd.to_datetime(features["date"].iloc[-1]).strftime("%Y-%m-%d")
    return {
        "model_dir": str(model_dir),
        "target_col": target_col,
        "tenor": tenor_label(target_col),
        "prediction_date": latest_date,
        "window": window,
        "prediction": LABEL_CN[pred_label],
        "prob_bearish": float(probs[0]),
        "prob_bullish": float(probs[1]),
        "prob_range": float(probs[2]),
        "detail": detail,
    }


def format_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def attention_bar(value: float, max_value: float) -> str:
    width = 0 if max_value <= 0 else max(2, min(100, value / max_value * 100))
    return f"<span class='bar'><i style='width:{width:.1f}%'></i></span>"


def sparkline(detail: pd.DataFrame, width: int = 300, height: int = 74) -> str:
    weights = detail.sort_values("date")["attention_weight"].astype(float).to_numpy()
    if len(weights) == 0:
        return ""
    pad = 8
    max_w = max(float(weights.max()), 1e-9)

    def x_pos(i: int) -> float:
        return pad + i * (width - 2 * pad) / max(1, len(weights) - 1)

    def y_pos(v: float) -> float:
        return height - pad - (v / max_w) * (height - 2 * pad)

    points = " ".join(f"{x_pos(i):.1f},{y_pos(float(v)):.1f}" for i, v in enumerate(weights))
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='注意力权重走势'>"
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#dbe3ea'/>"
        f"<polyline points='{points}' fill='none' stroke='#2f6f9f' stroke-width='2.4' "
        "stroke-linecap='round' stroke-linejoin='round'/>"
        "</svg>"
    )


def write_html(results: list[dict[str, Any]], path: Path, top_n: int) -> None:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    cards = []
    for result in results:
        detail = result["detail"].copy()
        top = detail.sort_values("attention_weight", ascending=False).head(top_n)
        max_weight = float(top["attention_weight"].max()) if not top.empty else 0.0
        top_rows = []
        for _, row in top.iterrows():
            yield_text = ""
            if "target_yield" in row and pd.notna(row["target_yield"]):
                yield_text = f"<td>{float(row['target_yield']):.4f}%</td>"
            else:
                yield_text = "<td>-</td>"
            top_rows.append(
                "<tr>"
                f"<td>{esc(row['date'])}</td>"
                f"<td>{int(row['days_before_latest'])}</td>"
                f"<td>{attention_bar(float(row['attention_weight']), max_weight)}</td>"
                f"<td>{format_pct(float(row['attention_weight']), 2)}</td>"
                f"{yield_text}"
                "</tr>"
            )

        pred_class = {
            "看空": "bear",
            "看多": "bull",
            "震荡": "range",
        }.get(result["prediction"], "range")
        cards.append(
            f"""
            <section class="card">
              <div class="card-head">
                <div>
                  <h2>{esc(result['tenor'])}</h2>
                  <p>预测日期：{esc(result['prediction_date'])}；窗口：最近 {int(result['window'])} 个交易日</p>
                </div>
                <strong class="pill {pred_class}">{esc(result['prediction'])}</strong>
              </div>
              <div class="prob-grid">
                <div><span>看空</span><b>{format_pct(result['prob_bearish'])}</b></div>
                <div><span>看多</span><b>{format_pct(result['prob_bullish'])}</b></div>
                <div><span>震荡</span><b>{format_pct(result['prob_range'])}</b></div>
              </div>
              <div class="spark">{sparkline(detail)}</div>
              <table>
                <thead><tr><th>模型重点看的日期</th><th>距最新日</th><th>权重</th><th>占比</th><th>该期限收益率</th></tr></thead>
                <tbody>{''.join(top_rows)}</tbody>
              </table>
              <p class="path">模型目录：{esc(result['model_dir'])}</p>
            </section>
            """
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LSTM 注意力解释报告</title>
<style>
:root {{
  --ink:#17202a; --muted:#667789; --line:#dce4ea; --paper:#fff; --back:#f5f7fa;
  --blue:#2f6f9f; --red:#c74343; --green:#238a62; --amber:#b7812f;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--back); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
header {{ background:var(--paper); border-bottom:1px solid var(--line); padding:30px 34px 22px; }}
main {{ width:min(1180px, calc(100% - 32px)); margin:22px auto 48px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
h1 {{ margin:0; font-size:30px; }}
header p {{ margin:8px 0 0; color:var(--muted); max-width:980px; }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:18px; }}
.card-head {{ display:flex; justify-content:space-between; gap:14px; align-items:flex-start; margin-bottom:14px; }}
h2 {{ margin:0; font-size:22px; }}
.card p {{ margin:4px 0 0; color:var(--muted); }}
.pill {{ padding:8px 18px; border-radius:999px; color:#fff; min-width:70px; text-align:center; }}
.bear {{ background:var(--red); }} .bull {{ background:var(--green); }} .range {{ background:var(--amber); }}
.prob-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:10px 0 14px; }}
.prob-grid div {{ border:1px solid var(--line); border-radius:6px; padding:10px; background:#fafbfd; }}
.prob-grid span {{ display:block; color:var(--muted); font-size:12px; }}
.prob-grid b {{ display:block; margin-top:2px; font-size:18px; }}
.spark {{ background:#fbfcfd; border:1px solid var(--line); border-radius:6px; padding:8px; margin-bottom:12px; }}
svg {{ width:100%; height:auto; display:block; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid var(--line); padding:8px 7px; text-align:left; white-space:nowrap; }}
th {{ color:var(--muted); background:#f6f8fa; }}
.bar {{ display:block; width:120px; height:9px; border-radius:999px; background:#e7edf2; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:var(--blue); border-radius:999px; }}
.path {{ overflow-wrap:anywhere; font-size:12px; }}
@media(max-width:860px) {{ main {{ grid-template-columns:1fr; }} h1 {{ font-size:26px; }} }}
</style>
</head>
<body>
<header>
  <h1>LSTM 注意力解释报告</h1>
  <p>生成时间：{esc(generated_at)}。这份报告回答一个很直白的问题：模型这次判断未来走势时，最近 60 个交易日里哪些日期被赋予了更高权重。权重不是因果证明，但可以帮助我们检查模型有没有盯着明显的市场转折、跳点或趋势段。</p>
</header>
<main>{''.join(cards)}</main>
</body>
</html>""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    features = load_features(args.features)

    model_dirs = [Path(path) for path in args.model_dir]
    if not model_dirs:
        model_dirs = latest_attention_models(args.rolling_summary)

    results = []
    detail_paths = []
    for model_dir in model_dirs:
        result = explain_model(model_dir, features, device)
        detail = result["detail"]
        detail_path = out_dir / f"{safe_name(result['tenor'])}_attention_detail.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        result["detail_csv"] = str(detail_path)
        detail_paths.append(str(detail_path))
        results.append(result)

    results.sort(key=lambda item: tenor_sort_key(item["target_col"]))
    summary = [
        {key: value for key, value in result.items() if key != "detail"}
        for result in results
    ]
    summary_csv = out_dir / "latest_attention_summary.csv"
    pd.DataFrame(summary).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json = out_dir / "latest_attention_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = out_dir / "latest_attention_report.html"
    write_html(results, html_path, args.top_n)

    print(
        json.dumps(
            {
                "HTML报告": str(html_path),
                "汇总CSV": str(summary_csv),
                "汇总JSON": str(summary_json),
                "明细CSV": detail_paths,
                "模型数量": len(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
