from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
LABELS = ["看空", "看多", "震荡"]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def report_date(report: dict[str, Any]) -> str:
    return str(report.get("data_end") or report.get("end_date") or report.get("run_date") or "")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_pct(value: Any, digits: int = 1) -> str:
    if not _finite(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_yield(value: Any) -> str:
    if not _finite(value):
        return "-"
    return f"{float(value):.4f}%"


def fmt_bp(value: Any) -> str:
    if not _finite(value):
        return "-"
    number = float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.1f} bp"


def fmt_prob(value: Any) -> str:
    return fmt_pct(value, 1)


def signal_tone(signal: str) -> str:
    text = str(signal)
    if "偏多" in text or "看多" in text:
        return "bullish"
    if "偏空" in text or "看空" in text:
        return "bearish"
    if "震荡" in text:
        return "range"
    return "neutral"


def signal_direction(signal: str) -> str:
    text = str(signal)
    if "偏多" in text or "看多" in text:
        return "看多"
    if "偏空" in text or "看空" in text:
        return "看空"
    return "震荡"


def tenor_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    tenors = (report.get("market_snapshot") or {}).get("tenors") or []
    rows: list[dict[str, Any]] = []
    for tenor in tenors:
        ensemble = tenor.get("ensemble_prediction") or {}
        probabilities = ensemble.get("probabilities") or {}
        signal = str(
            ensemble.get("display_prediction")
            or ensemble.get("threshold_prediction")
            or "震荡"
        )
        strict_signal = str(ensemble.get("threshold_prediction") or "震荡")
        rows.append(
            {
                "label": str(tenor.get("label") or ""),
                "latest_date": str(tenor.get("latest_date") or ""),
                "latest_yield": tenor.get("latest_yield"),
                "change_1d_bp": tenor.get("change_1d_bp"),
                "change_5d_bp": tenor.get("change_5d_bp"),
                "change_20d_bp": tenor.get("change_20d_bp"),
                "spread_latest_bp": tenor.get("spread_latest_bp"),
                "spread_change_20d_bp": tenor.get("spread_change_20d_bp"),
                "signal": signal,
                "strict_signal": strict_signal,
                "direction": str(ensemble.get("display_direction") or signal_direction(signal)),
                "strength": str(ensemble.get("display_strength") or ""),
                "model_count": int(ensemble.get("model_count") or 0),
                "prob_bearish": float(probabilities.get("看空", 0.0) or 0.0),
                "prob_bullish": float(probabilities.get("看多", 0.0) or 0.0),
                "prob_range": float(probabilities.get("震荡", 0.0) or 0.0),
                "note": str(tenor.get("note") or ""),
            }
        )
    return rows


def average_probs(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"看空": 0.0, "看多": 0.0, "震荡": 0.0}
    return {
        "看空": sum(row["prob_bearish"] for row in rows) / len(rows),
        "看多": sum(row["prob_bullish"] for row in rows) / len(rows),
        "震荡": sum(row["prob_range"] for row in rows) / len(rows),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = tenor_rows(report)
    if not rows:
        return {
            "direction": "暂无预测",
            "tone": "neutral",
            "plain": "本次日报没有可展示的期限模型结果。",
            "avg_probabilities": {"看空": 0.0, "看多": 0.0, "震荡": 0.0},
        }

    bullish = [row for row in rows if row["direction"] == "看多"]
    bearish = [row for row in rows if row["direction"] == "看空"]
    strong = [row for row in rows if row["strength"] == "强信号"]
    avg = average_probs(rows)

    if bullish and bearish:
        direction = "期限分化"
        tone = "range"
        plain = "不同期限的模型方向不一致，当前更适合按期限分别观察，而不是给出单边总判断。"
    elif bearish:
        direction = "综合震荡偏空" if not strong else "综合看空"
        tone = "bearish"
        labels = "、".join(row["label"] for row in bearish)
        plain = f"{labels}方向偏空，模型更担心对应期限信用债收益率上行；收益率上行通常意味着债券价格承压。"
    elif bullish:
        direction = "综合震荡偏多" if not strong else "综合看多"
        tone = "bullish"
        labels = "、".join(row["label"] for row in bullish)
        plain = f"{labels}方向偏多，模型更倾向对应期限信用债收益率下行；收益率下行通常有利于债券价格表现。"
    else:
        direction = "综合震荡"
        tone = "range"
        plain = "四个期限整体没有形成足够强的方向信号，当前更适合观察或维持纪律。"

    return {
        "direction": direction,
        "tone": tone,
        "plain": plain,
        "avg_probabilities": avg,
    }
