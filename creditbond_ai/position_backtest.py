from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import load_scaler
from .model import build_model


LABEL_TO_SIGNAL = {
    0: -1,
    1: 1,
    2: 0,
}

LABEL_CN = {
    0: "看空",
    1: "看多",
    2: "震荡",
}


@dataclass
class PositionBacktestConfig:
    features_path: str | Path
    model_dirs: list[str | Path]
    out_dir: str | Path
    initial_position: float = 0.5
    step: float = 0.05
    transaction_cost_bp: float = 0.0
    annual_days: int = 252
    include_carry: bool = True
    duration_map: dict[str, float] | None = None
    signal_scope: str = "test"


def _safe_name(text: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text, flags=re.UNICODE)
    return text.strip("_") or "model"


def _parse_duration_map(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    result: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"duration-map 片段缺少等号：{part}")
        key, value = part.split("=", 1)
        result[key.strip()] = float(value.strip())
    return result


def parse_duration_map(text: str | None) -> dict[str, float]:
    return _parse_duration_map(text)


def _load_checkpoint(model_dir: Path) -> dict[str, Any]:
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")
    return torch.load(model_path, map_location="cpu", weights_only=False)


def _infer_tenor(target_col: str) -> str:
    match = re.search(r"(\d+)\s*年", target_col)
    return match.group(1) if match else ""


def _infer_duration(target_col: str, checkpoint: dict[str, Any], duration_map: dict[str, float]) -> float:
    tenor = _infer_tenor(target_col)
    if tenor and tenor in duration_map:
        return float(duration_map[tenor])
    train_config = checkpoint.get("train_config", {})
    if isinstance(train_config, dict) and train_config.get("duration") is not None:
        return float(train_config["duration"])
    if tenor:
        return max(1.0, float(tenor) * 0.75)
    return 3.0


def _model_display_name(model_dir: Path, target_col: str) -> str:
    tenor = _infer_tenor(target_col)
    rating = "AAA+ " if "AAA+" in target_col else "AAA "
    model_type = model_dir.name
    if tenor:
        return f"{rating}{tenor}年 {model_type}"
    return f"{target_col} {model_type}"


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def _annualized_return(total_return: float, periods: int, annual_days: int) -> float:
    if periods <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (annual_days / periods) - 1.0)


def _sharpe(daily_returns: pd.Series, annual_days: int) -> float | None:
    if len(daily_returns) < 2:
        return None
    std = float(daily_returns.std(ddof=1))
    if std == 0:
        return None
    return float(daily_returns.mean() / std * math.sqrt(annual_days))


def _format_pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "不适用"
    return f"{float(value) * 100:.{digits}f}%"


def _format_num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "不适用"
    return f"{float(value):.{digits}f}"


def _clamp_position(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _prepare_asset_returns(
    features: pd.DataFrame,
    target_col: str,
    duration: float,
    annual_days: int,
    include_carry: bool,
) -> pd.DataFrame:
    if target_col not in features.columns:
        raise ValueError(f"特征表中找不到目标列：{target_col}")
    result = features[["date", target_col]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").dropna(subset=[target_col]).reset_index(drop=True)
    result["next_date"] = result["date"].shift(-1)
    result["next_yield"] = result[target_col].shift(-1)
    result["yield_change_1d"] = result["next_yield"] - result[target_col]
    result["price_return_proxy"] = -duration * (result["yield_change_1d"] / 100.0)
    if include_carry:
        result["carry_return_proxy"] = (result[target_col] / 100.0) / annual_days
    else:
        result["carry_return_proxy"] = 0.0
    result["bond_return_proxy"] = result["price_return_proxy"] + result["carry_return_proxy"]
    return result.dropna(subset=["next_date", "next_yield", "bond_return_proxy"]).copy()


def _load_test_predictions(model_dir: Path) -> pd.DataFrame:
    predictions_path = model_dir / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"找不到历史预测文件：{predictions_path}")
    pred = pd.read_csv(predictions_path, encoding="utf-8-sig")
    required = {"date", "y_pred", "y_pred_cn"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{predictions_path} 缺少字段：{sorted(missing)}")
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.sort_values("date").copy()
    pred["signal"] = pred["y_pred"].astype(int).map(LABEL_TO_SIGNAL).fillna(0).astype(int)
    return pred


def _predict_all_history(model_dir: Path, checkpoint: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(checkpoint.get("feature_cols", []))
    if not feature_cols:
        raise ValueError(f"模型缺少 feature_cols：{model_dir}")
    window = int(checkpoint.get("window", 0))
    if window < 2:
        raise ValueError(f"模型窗口长度异常：{model_dir}")
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise ValueError(f"特征表缺少模型需要的字段：{missing}")

    work = features[["date", *feature_cols]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any").reset_index(drop=True)
    if len(work) < window:
        raise ValueError(f"全历史回放至少需要 {window} 行，当前只有 {len(work)} 行。")

    scaler = load_scaler(model_dir / "scaler.joblib")
    x_scaled = scaler.transform(work[feature_cols].astype(float).to_numpy(dtype=np.float64)).astype(np.float32)
    windows = np.stack([x_scaled[i - window + 1 : i + 1] for i in range(window - 1, len(x_scaled))])

    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probs_chunks = []
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            batch = torch.from_numpy(windows[start : start + 512]).float()
            probs_chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probs = np.vstack(probs_chunks)
    pred_labels = probs.argmax(axis=1).astype(int)
    dates = work["date"].iloc[window - 1 :].reset_index(drop=True)
    pred = pd.DataFrame(
        {
            "date": dates,
            "y_pred": pred_labels,
            "y_pred_cn": [LABEL_CN[int(x)] for x in pred_labels],
            "prob_bearish": probs[:, 0],
            "prob_bullish": probs[:, 1],
            "prob_range": probs[:, 2],
        }
    )
    pred["signal"] = pred["y_pred"].map(LABEL_TO_SIGNAL).fillna(0).astype(int)
    return pred


def _run_one_backtest(
    model_dir: Path,
    features: pd.DataFrame,
    cfg: PositionBacktestConfig,
    duration_map: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    checkpoint = _load_checkpoint(model_dir)
    target_col = str(checkpoint.get("target_col", ""))
    if not target_col:
        raise ValueError(f"模型缺少 target_col：{model_dir}")
    duration = _infer_duration(target_col, checkpoint, duration_map)
    if cfg.signal_scope == "all":
        pred = _predict_all_history(model_dir, checkpoint, features)
    elif cfg.signal_scope == "test":
        pred = _load_test_predictions(model_dir)
    else:
        raise ValueError("signal_scope must be test or all")

    asset = _prepare_asset_returns(
        features=features,
        target_col=target_col,
        duration=duration,
        annual_days=cfg.annual_days,
        include_carry=cfg.include_carry,
    )
    merged = pred.merge(asset, on="date", how="inner")
    if merged.empty:
        raise ValueError(f"模型预测日期和特征表日期无法匹配：{model_dir}")

    position = _clamp_position(cfg.initial_position)
    value = 1.0
    buy_hold_value = 1.0
    neutral_value = 1.0
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        signal = int(row["signal"])
        before = position
        if signal > 0:
            position = _clamp_position(position + cfg.step)
        elif signal < 0:
            position = _clamp_position(position - cfg.step)
        trade = abs(position - before)
        cost_return = trade * (cfg.transaction_cost_bp / 10000.0)
        bond_return = float(row["bond_return_proxy"])
        strategy_return = position * bond_return - cost_return
        buy_hold_return = bond_return
        neutral_return = cfg.initial_position * bond_return
        value *= 1.0 + strategy_return
        buy_hold_value *= 1.0 + buy_hold_return
        neutral_value *= 1.0 + neutral_return
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "next_date": pd.Timestamp(row["next_date"]).date().isoformat(),
                "prediction": LABEL_CN.get(int(row["y_pred"]), str(row["y_pred_cn"])),
                "signal": signal,
                "position_before": before,
                "position_after": position,
                "trade": trade,
                "yield": float(row[target_col]),
                "next_yield": float(row["next_yield"]),
                "yield_change_1d": float(row["yield_change_1d"]),
                "price_return_proxy": float(row["price_return_proxy"]),
                "carry_return_proxy": float(row["carry_return_proxy"]),
                "bond_return_proxy": bond_return,
                "transaction_cost_return": cost_return,
                "strategy_return": strategy_return,
                "strategy_value": value,
                "buy_hold_return": buy_hold_return,
                "buy_hold_value": buy_hold_value,
                "neutral_return": neutral_return,
                "neutral_value": neutral_value,
            }
        )

    detail = pd.DataFrame(rows)
    periods = len(detail)
    total_return = float(detail["strategy_value"].iloc[-1] - 1.0)
    buy_hold_total_return = float(detail["buy_hold_value"].iloc[-1] - 1.0)
    neutral_total_return = float(detail["neutral_value"].iloc[-1] - 1.0)
    signal_counts = detail["prediction"].value_counts().to_dict()
    metrics = {
        "model_dir": str(model_dir),
        "display_name": _model_display_name(model_dir, target_col),
        "target_col": target_col,
        "signal_scope": cfg.signal_scope,
        "tenor": _infer_tenor(target_col),
        "duration_assumption": duration,
        "test_start": str(detail["date"].iloc[0]),
        "test_end": str(detail["next_date"].iloc[-1]),
        "signal_count": periods,
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, periods, cfg.annual_days),
        "max_drawdown": _max_drawdown(detail["strategy_value"]),
        "daily_volatility_annualized": float(detail["strategy_return"].std(ddof=1) * math.sqrt(cfg.annual_days)) if periods > 1 else 0.0,
        "sharpe": _sharpe(detail["strategy_return"], cfg.annual_days),
        "positive_day_ratio": float((detail["strategy_return"] > 0).mean()),
        "final_position": float(detail["position_after"].iloc[-1]),
        "average_position": float(detail["position_after"].mean()),
        "turnover": float(detail["trade"].sum()),
        "trade_count": int((detail["trade"] > 0).sum()),
        "signal_counts": {str(k): int(v) for k, v in signal_counts.items()},
        "buy_hold_total_return": buy_hold_total_return,
        "buy_hold_annualized_return": _annualized_return(buy_hold_total_return, periods, cfg.annual_days),
        "buy_hold_max_drawdown": _max_drawdown(detail["buy_hold_value"]),
        "neutral_total_return": neutral_total_return,
        "neutral_annualized_return": _annualized_return(neutral_total_return, periods, cfg.annual_days),
        "excess_vs_buy_hold": total_return - buy_hold_total_return,
        "excess_vs_neutral": total_return - neutral_total_return,
    }
    return metrics, detail


def _svg_line_chart(series: list[dict[str, Any]], keys: list[tuple[str, str, str]], width: int = 780, height: int = 260) -> str:
    if not series:
        return '<div class="empty-chart">暂无曲线数据</div>'
    pad_l, pad_r, pad_t, pad_b = 52, 20, 22, 38
    xs = list(range(len(series)))
    values: list[float] = []
    for key, _, _ in keys:
        values.extend(float(item[key]) for item in series if item.get(key) is not None and np.isfinite(float(item[key])))
    if not values:
        return '<div class="empty-chart">暂无曲线数据</div>'
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    else:
        span = y_max - y_min
        y_min -= span * 0.08
        y_max += span * 0.08

    def x_pos(i: int) -> float:
        if len(xs) == 1:
            return pad_l
        return pad_l + i * (width - pad_l - pad_r) / (len(xs) - 1)

    def y_pos(v: float) -> float:
        return pad_t + (y_max - v) * (height - pad_t - pad_b) / (y_max - y_min)

    polylines = []
    legends = []
    for key, label, color in keys:
        coords = []
        for i, item in enumerate(series):
            value = item.get(key)
            if value is None:
                continue
            coords.append(f"{x_pos(i):.1f},{y_pos(float(value)):.1f}")
        if coords:
            polylines.append(
                f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />'
            )
            legends.append(f'<span><i style="background:{color}"></i>{html.escape(label)}</span>')

    grid_values = [y_min, (y_min + y_max) / 2, y_max]
    grid = []
    for value in grid_values:
        y = y_pos(value)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e5ebef" stroke-width="1" />')
        grid.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#687582">{(value - 1) * 100:.1f}%</text>'
        )
    first_date = html.escape(str(series[0].get("date", "")))
    last_date = html.escape(str(series[-1].get("next_date", series[-1].get("date", ""))))
    return f"""
    <div class="chart-box">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="净值曲线">
        {''.join(grid)}
        <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#cfd8df" stroke-width="1" />
        {''.join(polylines)}
        <text x="{pad_l}" y="{height - 12}" font-size="12" fill="#687582">{first_date}</text>
        <text x="{width - pad_r}" y="{height - 12}" text-anchor="end" font-size="12" fill="#687582">{last_date}</text>
      </svg>
      <div class="legend">{''.join(legends)}</div>
    </div>
    """


def _svg_position_chart(series: list[dict[str, Any]], width: int = 780, height: int = 160) -> str:
    if not series:
        return '<div class="empty-chart">暂无仓位数据</div>'
    pad_l, pad_r, pad_t, pad_b = 44, 18, 16, 30

    def x_pos(i: int) -> float:
        if len(series) == 1:
            return pad_l
        return pad_l + i * (width - pad_l - pad_r) / (len(series) - 1)

    def y_pos(v: float) -> float:
        return pad_t + (1.0 - v) * (height - pad_t - pad_b)

    coords = [f"{x_pos(i):.1f},{y_pos(float(item['position_after'])):.1f}" for i, item in enumerate(series)]
    grid = []
    for value in [0.0, 0.5, 1.0]:
        y = y_pos(value)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e5ebef" stroke-width="1" />')
        grid.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#687582">{value * 100:.0f}%</text>')
    first_date = html.escape(str(series[0].get("date", "")))
    last_date = html.escape(str(series[-1].get("next_date", series[-1].get("date", ""))))
    return f"""
    <div class="chart-box small">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="仓位曲线">
        {''.join(grid)}
        <polyline points="{' '.join(coords)}" fill="none" stroke="#6b5aa6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
        <text x="{pad_l}" y="{height - 9}" font-size="12" fill="#687582">{first_date}</text>
        <text x="{width - pad_r}" y="{height - 9}" text-anchor="end" font-size="12" fill="#687582">{last_date}</text>
      </svg>
    </div>
    """


def _metrics_cards(metrics: dict[str, Any]) -> str:
    items = [
        ("累计收益率", _format_pct(metrics["total_return"])),
        ("年化收益率", _format_pct(metrics["annualized_return"])),
        ("最大回撤", _format_pct(metrics["max_drawdown"])),
        ("最终仓位", _format_pct(metrics["final_position"], 1)),
        ("换手倍数", _format_num(metrics["turnover"], 2)),
        ("胜率", _format_pct(metrics["positive_day_ratio"], 1)),
        ("买满基准", _format_pct(metrics["buy_hold_total_return"])),
        ("50%静态基准", _format_pct(metrics["neutral_total_return"])),
    ]
    return "\n".join(f"<div class=\"metric\"><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong></div>" for k, v in items)


def _summary_table(summary: list[dict[str, Any]]) -> str:
    rows = []
    for item in sorted(summary, key=lambda x: x["total_return"], reverse=True):
        rows.append(
            f"""
            <tr>
              <td>{html.escape(item["display_name"])}</td>
              <td>{html.escape(item["test_start"])} 至 {html.escape(item["test_end"])}</td>
              <td>{_format_pct(item["total_return"])}</td>
              <td>{_format_pct(item["annualized_return"])}</td>
              <td>{_format_pct(item["max_drawdown"])}</td>
              <td>{_format_pct(item["buy_hold_total_return"])}</td>
              <td>{_format_pct(item["final_position"], 1)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def _conclusion_cards(summary: list[dict[str, Any]]) -> str:
    if not summary:
        return ""
    best = max(summary, key=lambda x: x["total_return"])
    outperform_buy_hold = sum(1 for item in summary if item["total_return"] > item["buy_hold_total_return"])
    outperform_neutral = sum(1 for item in summary if item["total_return"] > item["neutral_total_return"])
    worst_drawdown = min(item["max_drawdown"] for item in summary)
    items = [
        (
            "策略最好的一段",
            f"{best['display_name']}，累计收益率 {_format_pct(best['total_return'])}，年化 {_format_pct(best['annualized_return'])}。",
        ),
        (
            "对比买满基准",
            f"{outperform_buy_hold}/{len(summary)} 个期限跑赢买满对应期限信用债。这个结果说明第一版规则更像控仓工具，不是进攻型策略。",
        ),
        (
            "对比50%静态仓位",
            f"{outperform_neutral}/{len(summary)} 个期限跑赢50%静态仓位。后续需要优化信号阈值和仓位步长。",
        ),
        (
            "最大回撤",
            f"{len(summary)} 组期限里最深回撤为 {_format_pct(worst_drawdown)}。样本很短，不能据此判断长期风险。",
        ),
    ]
    return "\n".join(
        f"""
        <div class="conclusion-card">
          <span>{html.escape(title)}</span>
          <strong>{html.escape(text)}</strong>
        </div>
        """
        for title, text in items
    )


def _rule_tenor_text(summary: list[dict[str, Any]]) -> str:
    tenors = []
    for item in summary:
        tenor = str(item.get("tenor", "")).strip()
        if tenor:
            tenors.append(tenor)
    unique = sorted(set(tenors), key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else 999)
    if not unique:
        return "按期限分别建仓：每个模型信号只调整对应期限仓位。"
    examples = "，".join(f"{tenor}年信号只调整{tenor}年仓位" for tenor in unique)
    return f"按期限分别建仓：{examples}。"


def _model_sections(results: list[dict[str, Any]]) -> str:
    blocks = []
    for result in results:
        metrics = result["metrics"]
        detail = result["detail"]
        records = detail.to_dict(orient="records")
        signal_counts = metrics.get("signal_counts", {})
        signal_text = "；".join(f"{k} {v}次" for k, v in signal_counts.items()) or "暂无"
        blocks.append(
            f"""
            <section class="section model-section">
              <div class="section-head">
                <div>
                  <h2>{html.escape(metrics["display_name"])}</h2>
                  <p class="muted">测试区间：{html.escape(metrics["test_start"])} 至 {html.escape(metrics["test_end"])}；久期假设：{_format_num(metrics["duration_assumption"], 1)}；信号：{html.escape(signal_text)}</p>
                </div>
              </div>
              <div class="metric-grid">
                {_metrics_cards(metrics)}
              </div>
              <h3>净值曲线</h3>
              {_svg_line_chart(records, [("strategy_value", "仓位规则策略", "#2f6f9f"), ("buy_hold_value", "买满信用债", "#87919b"), ("neutral_value", "50%静态仓位", "#b7812f")])}
              <h3>仓位变化</h3>
              {_svg_position_chart(records)}
            </section>
            """
        )
    return "\n".join(blocks)


def write_position_backtest_html(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    signal_scope = str(report["config"].get("signal_scope", "test"))
    scope_label = "全历史回放" if signal_scope == "all" else "样本外测试集回测"
    scope_note = (
        "本页使用已训练模型对全部可用历史逐日重新生成信号，包含训练期和验证期，只适合观察策略形态。"
        if signal_scope == "all"
        else "本页只使用训练时留出的测试集信号，更适合观察样本外表现。"
    )
    best = max(summary, key=lambda x: x["total_return"]) if summary else None
    headline = "暂无可用回测"
    if best:
        headline = (
            f"本次{scope_label}下，{best['display_name']} 累计收益率最高，"
            f"为 {_format_pct(best['total_return'])}。"
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 仓位规则回测</title>
  <style>
    :root {{
      --ink: #18202a;
      --muted: #64717f;
      --paper: #ffffff;
      --back: #f4f6f8;
      --line: #dce4ea;
      --blue: #2f6f9f;
      --green: #238a62;
      --red: #c74343;
      --amber: #b7812f;
      --violet: #6b5aa6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--back);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    .top {{
      background: var(--paper);
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
    }}
    .hero {{
      padding: 34px 0 26px;
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 24px;
      align-items: end;
    }}
    .eyebrow {{
      color: var(--blue);
      font-weight: 800;
      font-size: 14px;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      letter-spacing: 0;
    }}
    .hero p {{
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 760px;
    }}
    .rule-box {{
      border: 1px solid var(--line);
      background: #fbfcfd;
      border-radius: 8px;
      padding: 16px;
    }}
    .rule-box strong {{
      display: block;
      font-size: 20px;
      margin-bottom: 8px;
    }}
    .main {{
      display: grid;
      gap: 18px;
      padding: 22px 0 46px;
    }}
    .section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 21px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 20px 0 10px;
      font-size: 16px;
    }}
    .muted {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .conclusion-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .conclusion-card {{
      background: #f8fafb;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .conclusion-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .conclusion-card strong {{
      display: block;
      font-size: 15px;
      font-weight: 700;
    }}
    .metric {{
      background: #f8fafb;
      border-top: 3px solid var(--blue);
      border-radius: 6px;
      padding: 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
      font-variant-numeric: tabular-nums;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .summary-table th,
    .summary-table td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: top;
      font-variant-numeric: tabular-nums;
    }}
    .summary-table th:first-child,
    .summary-table td:first-child,
    .summary-table th:nth-child(2),
    .summary-table td:nth-child(2) {{
      text-align: left;
    }}
    .summary-table th {{
      color: var(--muted);
      font-weight: 700;
      background: #f8fafb;
    }}
    .note {{
      border-left: 4px solid var(--amber);
      background: #fff9ef;
      color: #4b3a23;
      padding: 12px 14px;
      border-radius: 6px;
    }}
    .chart-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }}
    .chart-box svg {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
      border-radius: 6px;
    }}
    .chart-box.small svg {{
      max-height: 180px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 18px;
      height: 3px;
      display: inline-block;
      border-radius: 999px;
    }}
    .file-list {{
      display: grid;
      gap: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 14px;
    }}
    @media (max-width: 860px) {{
      .hero, .metric-grid {{
        grid-template-columns: 1fr;
      }}
      .conclusion-grid {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 28px;
      }}
      .section {{
        padding: 18px;
      }}
      .summary-table {{
        display: block;
        overflow-x: auto;
        white-space: nowrap;
      }}
    }}
  </style>
</head>
<body>
  <header class="top">
    <div class="wrap hero">
      <div>
        <div class="eyebrow">信用债 AI 仓位规则回测</div>
        <h1>{html.escape(headline)}</h1>
        <p>生成日期：{html.escape(str(report["generated_at"]))}。{html.escape(scope_note)}本页模拟一个透明的仓位加减规则。</p>
      </div>
      <div class="rule-box">
        <strong>规则</strong>
        <p class="muted">{html.escape(_rule_tenor_text(summary))}初始仓位 {html.escape(_format_pct(report["config"]["initial_position"], 1))}；看多加 {html.escape(_format_pct(report["config"]["step"], 1))} 该期限仓位；看空减 {html.escape(_format_pct(report["config"]["step"], 1))} 该期限仓位；震荡不动；单个期限仓位限制在 0% 至 100%。</p>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>一句话结论</h2>
      <p class="muted">这版规则能把神经网络信号翻译成期限仓位动作，但目前更适合作为研究辅助和控仓参考。</p>
      <div class="conclusion-grid">
        {_conclusion_cards(summary)}
      </div>
    </section>

    <section class="section">
      <h2>结果总览</h2>
      <table class="summary-table">
        <thead>
          <tr>
            <th>模型</th>
            <th>回测区间</th>
            <th>累计收益率</th>
            <th>年化收益率</th>
            <th>最大回撤</th>
            <th>买满基准</th>
            <th>最终仓位</th>
          </tr>
        </thead>
        <tbody>
          {_summary_table(summary)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>重要说明</h2>
      <div class="note">这是第一版简化回测。收益用“票息 carry + 久期近似价格变化”估算，不是中债财富指数的真实全价收益；样本来自目前 DM 可取到的短历史，年化收益率只是机械换算，不能过度解读。当前口径：{html.escape(scope_label)}。</div>
    </section>

    {_model_sections(report["results"])}

    <section class="section">
      <h2>文件</h2>
      <div class="file-list">
        <div>特征表：{html.escape(str(report["features_path"]))}</div>
        <div>汇总 CSV：{html.escape(str(report["summary_csv"]))}</div>
        <div>结构化 JSON：{html.escape(str(report["json_path"]))}</div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def run_position_backtest(cfg: PositionBacktestConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(cfg.features_path)
    features = pd.read_csv(features_path, encoding="utf-8-sig")
    if "date" not in features.columns:
        raise ValueError("特征表必须包含 date 列。")
    duration_map = cfg.duration_map or {}

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_dir_raw in cfg.model_dirs:
        model_dir = Path(model_dir_raw)
        metrics, detail = _run_one_backtest(model_dir, features, cfg, duration_map)
        safe = _safe_name(metrics["display_name"])
        detail_path = out_dir / f"{safe}_position_backtest.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        metrics["detail_csv"] = str(detail_path)
        results.append({"metrics": metrics, "detail": detail})
        summary_rows.append(metrics.copy())

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "position_backtest_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    json_path = out_dir / "position_backtest_report.json"
    html_path = out_dir / "position_backtest_report.html"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_path": str(features_path),
        "config": {
            "initial_position": cfg.initial_position,
            "step": cfg.step,
            "transaction_cost_bp": cfg.transaction_cost_bp,
            "annual_days": cfg.annual_days,
            "include_carry": cfg.include_carry,
            "duration_map": duration_map,
            "signal_scope": cfg.signal_scope,
        },
        "summary": summary_rows,
        "results": [{"metrics": item["metrics"], "detail": item["detail"]} for item in results],
        "summary_csv": str(summary_csv),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }

    json_safe = {
        **report,
        "results": [
            {
                "metrics": item["metrics"],
                "detail_rows": item["detail"].to_dict(orient="records"),
            }
            for item in results
        ],
    }
    json_path.write_text(json.dumps(json_safe, ensure_ascii=False, indent=2), encoding="utf-8")
    write_position_backtest_html(report, html_path)
    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary_csv": str(summary_csv),
        "summary": summary_rows,
    }
