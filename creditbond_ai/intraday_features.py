from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TENOR_ORDER = {"2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
ACTIVE_TENOR_MAP = {
    "2YTBOND": "2Y",
    "5YTBOND": "5Y",
    "10YTBOND": "10Y",
    "30YTBOND": "30Y",
}
FUTURES_TENOR_MAP = {
    "TS": "2Y",
    "TF": "5Y",
    "T": "10Y",
    "TL": "30Y",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("*", "", regex=False), errors="coerce")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _fmt_pct(value: Any, digits: int = 1) -> str:
    number = _safe_float(value, math.nan)
    if not np.isfinite(number):
        return "-"
    return f"{number * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, math.nan)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _fmt_signed(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _safe_float(value, math.nan)
    if not np.isfinite(number):
        return "-"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.{digits}f}{suffix}"


def _basis_symbol(code: str) -> str:
    text = str(code)
    if text.startswith("TL"):
        return "TL"
    if text.startswith("TF"):
        return "TF"
    if text.startswith("TS"):
        return "TS"
    if text.startswith("T"):
        return "T"
    return text


def _yield_signal(change_bp: float) -> str:
    if change_bp >= 1.0:
        return "利率上行"
    if change_bp <= -1.0:
        return "利率下行"
    return "窄幅波动"


def _pressure_signal(score: float) -> str:
    if score >= 1.0:
        return "利率压力升温"
    if score <= -1.0:
        return "利率环境友好"
    return "利率中性震荡"


def _quote_signal(width_bp: float) -> str:
    if width_bp >= 0.8:
        return "报价分歧偏宽"
    if width_bp <= 0.25:
        return "报价分歧较窄"
    return "报价分歧正常"


def _sentiment_signal(score: float) -> str:
    if score >= 5:
        return "机构指数偏强"
    if score <= -5:
        return "机构指数偏弱"
    return "机构指数中性"


def build_bond_bar_features(bars: pd.DataFrame, rolling: pd.DataFrame | None = None) -> pd.DataFrame:
    if bars.empty or "securityId" not in bars.columns:
        return pd.DataFrame()
    work = bars.copy()
    work["issueTime"] = work.get("issueTime", "").astype(str)
    for col in [
        "openYield",
        "closeYield",
        "highYield",
        "lowYield",
        "ytdCloseYield",
        "tradeNum",
        "tknTradeNum",
        "gvnTradeNum",
        "trdTradeNum",
    ]:
        if col in work.columns:
            work[col] = _num(work[col])
    term_by_code: dict[str, str] = {}
    if rolling is not None and not rolling.empty:
        key_col = "keyTenor" if "keyTenor" in rolling.columns else ""
        code_col = "bondCode" if "bondCode" in rolling.columns else ""
        if key_col and code_col:
            for _, row in rolling.iterrows():
                term_by_code[str(row[code_col])] = f"{int(float(row[key_col]))}Y"

    rows: list[dict[str, Any]] = []
    for security_id, group in work.groupby("securityId", dropna=False):
        group = group.sort_values("issueTime")
        first = group.iloc[0]
        last = group.iloc[-1]
        open_yield = _safe_float(first.get("openYield"), _safe_float(first.get("closeYield"), math.nan))
        latest_yield = _safe_float(last.get("closeYield"), math.nan)
        ytd_close = _safe_float(last.get("ytdCloseYield"), math.nan)
        high_yield = _safe_float(group.get("highYield", pd.Series(dtype=float)).max(), math.nan)
        low_yield = _safe_float(group.get("lowYield", pd.Series(dtype=float)).min(), math.nan)
        term = ACTIVE_TENOR_MAP.get(str(security_id), term_by_code.get(str(security_id), ""))
        rows.append(
            {
                "security_id": str(security_id),
                "sec_short_name": str(last.get("secShortName", "")),
                "tenor": term,
                "first_time": str(first.get("issueTime", "")),
                "latest_time": str(last.get("issueTime", "")),
                "open_yield": open_yield,
                "latest_yield": latest_yield,
                "yield_change_bp": (latest_yield - open_yield) * 100 if np.isfinite(open_yield + latest_yield) else np.nan,
                "ytd_change_bp": (latest_yield - ytd_close) * 100 if np.isfinite(latest_yield + ytd_close) else np.nan,
                "intraday_range_bp": (high_yield - low_yield) * 100 if np.isfinite(high_yield + low_yield) else np.nan,
                "trade_num": int(_safe_float(group.get("tradeNum", pd.Series(dtype=float)).sum())),
                "tkn_trade_num": int(_safe_float(group.get("tknTradeNum", pd.Series(dtype=float)).sum())),
                "gvn_trade_num": int(_safe_float(group.get("gvnTradeNum", pd.Series(dtype=float)).sum())),
                "trd_trade_num": int(_safe_float(group.get("trdTradeNum", pd.Series(dtype=float)).sum())),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["signal"] = out["yield_change_bp"].map(_yield_signal)
        out["tenor_order"] = out["tenor"].map(TENOR_ORDER).fillna(999)
        out = out.sort_values(["tenor_order", "security_id"]).drop(columns=["tenor_order"]).reset_index(drop=True)
    return out


def build_realtime_quote_features(quotes: pd.DataFrame, rolling: pd.DataFrame | None = None) -> pd.DataFrame:
    if quotes.empty or "securityId" not in quotes.columns:
        return pd.DataFrame()
    work = quotes.copy()
    term_by_code: dict[str, str] = {}
    if rolling is not None and not rolling.empty and "bondCode" in rolling.columns and "keyTenor" in rolling.columns:
        for _, row in rolling.iterrows():
            term_by_code[str(row["bondCode"])] = f"{int(float(row['keyTenor']))}Y"
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        bid_yield = _safe_float(row.get("brokerBidYield"), math.nan)
        ofr_yield = _safe_float(row.get("brokerOfrYield"), math.nan)
        bid_ofr_bp = _safe_float(row.get("brokerBidYieldSubOfr"), math.nan) * 100
        trade_ytd_bp = _safe_float(row.get("brokerTradeYieldSubYtdClose"), math.nan) * 100
        rows.append(
            {
                "security_id": str(row.get("securityId", "")),
                "sec_short_name": str(row.get("secShortName", "")),
                "tenor": term_by_code.get(str(row.get("securityId", "")), ""),
                "broker_issue_time": str(row.get("brokerIssueTime", "")),
                "broker_trade_time": str(row.get("brokerTradeTime", "")),
                "broker_bid_yield": bid_yield,
                "broker_ofr_yield": ofr_yield,
                "broker_bid_ofr_width_bp": bid_ofr_bp if np.isfinite(bid_ofr_bp) else (bid_yield - ofr_yield) * 100,
                "broker_trade_ytd_change_bp": trade_ytd_bp,
                "broker_trade_num": int(_safe_float(row.get("brokerTradeNum"))),
                "xbond_bid_yield": _safe_float(row.get("xbondBidYield"), math.nan),
                "xbond_ofr_yield": _safe_float(row.get("xbondOfrYield"), math.nan),
                "twoside_bid_yield": _safe_float(row.get("twosideBidYield"), math.nan),
                "twoside_ofr_yield": _safe_float(row.get("twosideOfrYield"), math.nan),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["quote_signal"] = out["broker_bid_ofr_width_bp"].map(_quote_signal)
        out["tenor_order"] = out["tenor"].map(TENOR_ORDER).fillna(999)
        out = out.sort_values(["tenor_order", "security_id"]).drop(columns=["tenor_order"]).reset_index(drop=True)
    return out


def build_futures_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or "securityId" not in bars.columns:
        return pd.DataFrame()
    work = bars.copy()
    work["issueTime"] = work.get("issueTime", "").astype(str)
    for col in ["openPrice", "closePrice", "highPrice", "lowPrice", "openInterest", "oiChange", "volumeChange"]:
        if col in work.columns:
            work[col] = _num(work[col])
    rows: list[dict[str, Any]] = []
    for security_id, group in work.groupby("securityId", dropna=False):
        group = group.sort_values("issueTime")
        first = group.iloc[0]
        last = group.iloc[-1]
        open_price = _safe_float(first.get("openPrice"), _safe_float(first.get("closePrice"), math.nan))
        latest_price = _safe_float(last.get("closePrice"), math.nan)
        symbol = _basis_symbol(str(security_id))
        rows.append(
            {
                "security_id": str(security_id),
                "symbol": symbol,
                "tenor": FUTURES_TENOR_MAP.get(symbol, ""),
                "first_time": str(first.get("issueTime", "")),
                "latest_time": str(last.get("issueTime", "")),
                "open_price": open_price,
                "latest_price": latest_price,
                "price_change": latest_price - open_price if np.isfinite(open_price + latest_price) else np.nan,
                "price_change_pct": latest_price / open_price - 1 if open_price and np.isfinite(open_price + latest_price) else np.nan,
                "intraday_range": _safe_float(group.get("highPrice", pd.Series(dtype=float)).max(), math.nan)
                - _safe_float(group.get("lowPrice", pd.Series(dtype=float)).min(), math.nan),
                "latest_open_interest": _safe_float(last.get("openInterest"), math.nan),
                "volume_change": _safe_float(group.get("volumeChange", pd.Series(dtype=float)).sum(), math.nan),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bond_signal"] = np.where(out["price_change"] > 0, "期货偏强", np.where(out["price_change"] < 0, "期货偏弱", "期货震荡"))
        out["tenor_order"] = out["tenor"].map(TENOR_ORDER).fillna(999)
        out = out.sort_values(["tenor_order", "security_id"]).drop(columns=["tenor_order"]).reset_index(drop=True)
    return out


def build_basis_features(basis: pd.DataFrame) -> pd.DataFrame:
    if basis.empty or "securityId" not in basis.columns:
        return pd.DataFrame()
    work = basis.copy()
    rows: list[dict[str, Any]] = []
    for security_id, group in work.groupby("securityId", dropna=False):
        group = group.copy()
        group["_is_ctd"] = group.get("bondCode", "").astype(str) == group.get("ctdBondCode", "").astype(str)
        preferred = group[group["_is_ctd"]]
        if preferred.empty:
            preferred = group.dropna(subset=["tradeYield"], how="all")
        if preferred.empty:
            preferred = group
        row = preferred.iloc[0]
        symbol = _basis_symbol(str(security_id))
        rows.append(
            {
                "security_id": str(security_id),
                "symbol": symbol,
                "tenor": FUTURES_TENOR_MAP.get(symbol, ""),
                "bond_code": str(row.get("bondCode", "")),
                "ctd_bond_code": str(row.get("ctdBondCode", "")),
                "trade_time": str(row.get("tradeTime", "")),
                "trade_yield": _safe_float(row.get("tradeYield"), math.nan),
                "trade_num": int(_safe_float(row.get("tradeNum"))),
                "close_basis": _safe_float(row.get("closeBasis"), math.nan),
                "close_irr": _safe_float(row.get("closeIrr"), math.nan),
                "close_bnoc": _safe_float(row.get("closeBnoc"), math.nan),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["tenor_order"] = out["tenor"].map(TENOR_ORDER).fillna(999)
        out = out.sort_values(["tenor_order", "security_id"]).drop(columns=["tenor_order"]).reset_index(drop=True)
    return out


def build_sentiment_features(sentiments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_cols = ["indexBank", "indexSecurities", "indexInsurance", "indexFund", "indexOthers"]
    for name, df in sentiments.items():
        if df.empty:
            continue
        work = df.copy()
        work["issueTime"] = work.get("issueTime", "").astype(str)
        for col in numeric_cols:
            if col in work.columns:
                work[col] = _num(work[col])
        for (index_type, index_desc), group in work.groupby(["indexType", "indexTypeDesc"], dropna=False):
            group = group.sort_values("issueTime")
            last = group.iloc[-1]
            latest_values = [_safe_float(last.get(col), math.nan) for col in numeric_cols if col in group.columns]
            latest_score = float(np.nanmean(latest_values)) if latest_values else np.nan
            rows.append(
                {
                    "source": name,
                    "data_source_desc": str(last.get("dataSourceDesc", "")),
                    "index_type": str(index_type),
                    "index_type_desc": str(index_desc),
                    "latest_time": str(last.get("issueTime", "")),
                    "latest_score": latest_score,
                    "latest_bank": _safe_float(last.get("indexBank"), math.nan),
                    "latest_securities": _safe_float(last.get("indexSecurities"), math.nan),
                    "latest_insurance": _safe_float(last.get("indexInsurance"), math.nan),
                    "latest_fund": _safe_float(last.get("indexFund"), math.nan),
                    "latest_others": _safe_float(last.get("indexOthers"), math.nan),
                    "rows": int(len(group)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sentiment_signal"] = out["latest_score"].map(_sentiment_signal)
        out = out.sort_values(["data_source_desc", "index_type_desc"]).reset_index(drop=True)
    return out


def build_feature_vector(
    snapshot_date: str,
    bond_features: pd.DataFrame,
    quote_features: pd.DataFrame,
    futures_features: pd.DataFrame,
    basis_features: pd.DataFrame,
    sentiment_features: pd.DataFrame,
) -> pd.DataFrame:
    row: dict[str, Any] = {"snapshot_date": snapshot_date}
    if not bond_features.empty:
        for _, item in bond_features.iterrows():
            tenor = str(item.get("tenor") or item.get("security_id", "")).lower()
            row[f"tbond_{tenor}_yield_change_bp"] = item.get("yield_change_bp")
            row[f"tbond_{tenor}_ytd_change_bp"] = item.get("ytd_change_bp")
            row[f"tbond_{tenor}_trade_num"] = item.get("trade_num")
        row["tbond_avg_yield_change_bp"] = float(pd.to_numeric(bond_features["yield_change_bp"], errors="coerce").mean())
        row["tbond_total_trade_num"] = int(pd.to_numeric(bond_features["trade_num"], errors="coerce").fillna(0).sum())
    if not quote_features.empty:
        row["quote_avg_bid_ofr_width_bp"] = float(pd.to_numeric(quote_features["broker_bid_ofr_width_bp"], errors="coerce").mean())
        row["quote_max_bid_ofr_width_bp"] = float(pd.to_numeric(quote_features["broker_bid_ofr_width_bp"], errors="coerce").max())
    if not futures_features.empty:
        for _, item in futures_features.iterrows():
            symbol = str(item.get("symbol", "")).lower()
            row[f"future_{symbol}_price_change"] = item.get("price_change")
            row[f"future_{symbol}_price_change_pct"] = item.get("price_change_pct")
            row[f"future_{symbol}_volume_change"] = item.get("volume_change")
        row["future_avg_price_change_pct"] = float(pd.to_numeric(futures_features["price_change_pct"], errors="coerce").mean())
    if not basis_features.empty:
        for _, item in basis_features.iterrows():
            symbol = str(item.get("symbol", "")).lower()
            row[f"basis_{symbol}_close_irr"] = item.get("close_irr")
            row[f"basis_{symbol}_close_basis"] = item.get("close_basis")
    if not sentiment_features.empty:
        row["sentiment_avg_latest_score"] = float(pd.to_numeric(sentiment_features["latest_score"], errors="coerce").mean())
        row["sentiment_min_latest_score"] = float(pd.to_numeric(sentiment_features["latest_score"], errors="coerce").min())
        row["sentiment_max_latest_score"] = float(pd.to_numeric(sentiment_features["latest_score"], errors="coerce").max())
    return pd.DataFrame([row])


def build_intraday_features(snapshot_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    snapshot_dir = Path(snapshot_dir)
    out_dir = Path(out_dir) if out_dir else snapshot_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_date = snapshot_dir.name

    rolling = _read_csv(snapshot_dir / "tbond_rolling_active.csv")
    bond_bars = _read_csv(snapshot_dir / "tbond_active_bars.csv")
    actual_bars = _read_csv(snapshot_dir / "tbond_actual_bars.csv")
    quotes = _read_csv(snapshot_dir / "tbond_realtime_quote.csv")
    futures_bars = _read_csv(snapshot_dir / "treasury_futures_bars.csv")
    basis = _read_csv(snapshot_dir / "treasury_futures_basis.csv")
    sentiments = {
        "国际货币": _read_csv(snapshot_dir / "bond_insti_sentiment_10.csv"),
        "国利货币": _read_csv(snapshot_dir / "bond_insti_sentiment_14.csv"),
    }

    bond_features = build_bond_bar_features(bond_bars, rolling=rolling)
    actual_bond_features = build_bond_bar_features(actual_bars, rolling=rolling)
    quote_features = build_realtime_quote_features(quotes, rolling=rolling)
    futures_features = build_futures_features(futures_bars)
    basis_features = build_basis_features(basis)
    sentiment_features = build_sentiment_features(sentiments)
    feature_vector = build_feature_vector(
        snapshot_date=snapshot_date,
        bond_features=bond_features,
        quote_features=quote_features,
        futures_features=futures_features,
        basis_features=basis_features,
        sentiment_features=sentiment_features,
    )

    outputs = {
        "bond_features": out_dir / "tbond_active_features.csv",
        "actual_bond_features": out_dir / "tbond_actual_features.csv",
        "quote_features": out_dir / "tbond_realtime_quote_features.csv",
        "futures_features": out_dir / "treasury_futures_features.csv",
        "basis_features": out_dir / "treasury_futures_basis_features.csv",
        "sentiment_features": out_dir / "bond_insti_sentiment_features.csv",
        "feature_vector": out_dir / "intraday_feature_vector.csv",
    }
    for key, path in outputs.items():
        locals()[key].to_csv(path, index=False, encoding="utf-8-sig")
    return outputs


def _bar_points(values: list[float], width: int = 260, height: int = 58) -> str:
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return ""
    lo, hi = min(finite), max(finite)
    span = hi - lo or 1.0
    points = []
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        x = idx * width / max(1, len(values) - 1)
        y = height - (value - lo) * height / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _sparkline(df: pd.DataFrame, security_id: str, value_col: str) -> str:
    if df.empty or "securityId" not in df.columns or value_col not in df.columns:
        return ""
    group = df[df["securityId"].astype(str) == str(security_id)].sort_values("issueTime")
    values = pd.to_numeric(group[value_col], errors="coerce").tolist()
    points = _bar_points(values)
    if not points:
        return ""
    return f'<svg viewBox="0 0 260 58" role="img"><polyline points="{points}" fill="none" stroke="#2e6f9e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def render_intraday_radar(snapshot_dir: str | Path, features_dir: str | Path | None = None, out_path: str | Path | None = None) -> Path:
    snapshot_dir = Path(snapshot_dir)
    features_dir = Path(features_dir) if features_dir else snapshot_dir / "features"
    out_path = Path(out_path) if out_path else snapshot_dir / "reports" / "intraday_radar_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_bond_bars = _read_csv(snapshot_dir / "tbond_active_bars.csv")
    raw_futures_bars = _read_csv(snapshot_dir / "treasury_futures_bars.csv")
    bond_features = _read_csv(features_dir / "tbond_active_features.csv")
    quote_features = _read_csv(features_dir / "tbond_realtime_quote_features.csv")
    futures_features = _read_csv(features_dir / "treasury_futures_features.csv")
    basis_features = _read_csv(features_dir / "treasury_futures_basis_features.csv")
    sentiment_features = _read_csv(features_dir / "bond_insti_sentiment_features.csv")
    vector = _read_csv(features_dir / "intraday_feature_vector.csv")
    vector_row = vector.iloc[0].to_dict() if not vector.empty else {}

    avg_yield_change = _safe_float(vector_row.get("tbond_avg_yield_change_bp"), math.nan)
    avg_future_pct = _safe_float(vector_row.get("future_avg_price_change_pct"), math.nan)
    avg_quote_width = _safe_float(vector_row.get("quote_avg_bid_ofr_width_bp"), math.nan)
    avg_sentiment = _safe_float(vector_row.get("sentiment_avg_latest_score"), math.nan)
    rate_score = avg_yield_change - avg_future_pct * 1000 if np.isfinite(avg_yield_change + avg_future_pct) else avg_yield_change
    overall = _pressure_signal(rate_score)

    bond_cards = []
    for _, row in bond_features.iterrows():
        spark = _sparkline(raw_bond_bars, str(row.get("security_id")), "closeYield")
        bond_cards.append(
            f"""
            <article class="card">
              <div class="card-head"><h3>{row.get('tenor') or row.get('security_id')}</h3><span class="pill">{row.get('signal')}</span></div>
              <p class="muted">{row.get('sec_short_name')} ｜ {row.get('first_time')} 至 {row.get('latest_time')}</p>
              <div class="spark">{spark}</div>
              <div class="mini-grid">
                <div><span>最新收益率</span><b>{_fmt_num(row.get('latest_yield'), 4)}%</b></div>
                <div><span>日内变化</span><b>{_fmt_signed(row.get('yield_change_bp'), 2, ' bp')}</b></div>
                <div><span>较昨收</span><b>{_fmt_signed(row.get('ytd_change_bp'), 2, ' bp')}</b></div>
                <div><span>成交笔数</span><b>{int(_safe_float(row.get('trade_num')))}</b></div>
              </div>
            </article>
            """
        )

    futures_rows = []
    for _, row in futures_features.iterrows():
        basis_row = basis_features[basis_features["security_id"].astype(str) == str(row.get("security_id"))]
        basis_text = "-"
        irr_text = "-"
        if not basis_row.empty:
            basis_text = _fmt_num(basis_row.iloc[0].get("close_basis"), 3)
            irr_text = _fmt_num(basis_row.iloc[0].get("close_irr"), 2) + "%"
        futures_rows.append(
            f"""
            <tr>
              <td>{row.get('security_id')}</td>
              <td>{row.get('tenor')}</td>
              <td>{row.get('latest_time')}</td>
              <td>{_fmt_num(row.get('latest_price'), 3)}</td>
              <td>{_fmt_signed(row.get('price_change'), 3)}</td>
              <td>{_fmt_pct(row.get('price_change_pct'), 2)}</td>
              <td>{basis_text}</td>
              <td>{irr_text}</td>
              <td>{_fmt_num(row.get('volume_change'), 0)}</td>
            </tr>
            """
        )

    quote_rows = []
    for _, row in quote_features.iterrows():
        quote_rows.append(
            f"""
            <tr>
              <td>{row.get('tenor')}</td>
              <td>{row.get('security_id')}</td>
              <td>{row.get('broker_issue_time')}</td>
              <td>{_fmt_num(row.get('broker_bid_yield'), 4)}%</td>
              <td>{_fmt_num(row.get('broker_ofr_yield'), 4)}%</td>
              <td>{_fmt_num(row.get('broker_bid_ofr_width_bp'), 2)} bp</td>
              <td>{_fmt_signed(row.get('broker_trade_ytd_change_bp'), 2, ' bp')}</td>
              <td>{row.get('quote_signal')}</td>
            </tr>
            """
        )

    sentiment_rows = []
    for _, row in sentiment_features.head(18).iterrows():
        sentiment_rows.append(
            f"""
            <tr>
              <td>{row.get('data_source_desc')}</td>
              <td>{row.get('index_type_desc')}</td>
              <td>{row.get('latest_time')}</td>
              <td>{_fmt_num(row.get('latest_score'), 1)}</td>
              <td>{_fmt_num(row.get('latest_bank'), 1)}</td>
              <td>{_fmt_num(row.get('latest_securities'), 1)}</td>
              <td>{_fmt_num(row.get('latest_fund'), 1)}</td>
              <td>{row.get('sentiment_signal')}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>信用债 AI 日内雷达：{snapshot_dir.name}</title>
  <style>
    :root {{ --ink:#17202c; --muted:#607086; --line:#dbe4ee; --bg:#eef3f8; --paper:#fff; --red:#d64b4b; --green:#248461; --gold:#b9812a; --blue:#2e6f9e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--ink); }}
    .wrap {{ width:min(1220px, calc(100% - 36px)); margin:0 auto; }}
    .hero {{ padding:30px 0 20px; background:linear-gradient(180deg,#fbfcff,#edf3f8); border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--blue); font-weight:800; }}
    h1 {{ margin:8px 0 8px; font-size:36px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:22px; }}
    h3 {{ margin:0; font-size:22px; }}
    p {{ line-height:1.7; }}
    .muted {{ color:var(--muted); }}
    .main {{ display:grid; gap:16px; padding:18px 0 42px; }}
    .section,.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 10px 24px rgba(35,48,66,.05); }}
    .section {{ padding:18px; }}
    .signal-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:14px; }}
    .metric {{ background:#f6f8fb; border-top:3px solid var(--blue); border-radius:6px; padding:10px 12px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric b {{ display:block; margin-top:5px; font-size:18px; }}
    .card-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ padding:15px; }}
    .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; }}
    .pill {{ display:inline-block; border-radius:999px; padding:5px 9px; color:#fff; background:var(--gold); font-weight:800; font-size:12px; }}
    .spark svg {{ width:100%; height:58px; margin:8px 0; background:#f8fafc; border-radius:6px; }}
    .mini-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .mini-grid div {{ background:#f6f8fb; border-radius:6px; padding:8px; }}
    .mini-grid span {{ display:block; color:var(--muted); font-size:11px; }}
    .mini-grid b {{ display:block; margin-top:4px; font-size:14px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:880px; background:#fff; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid #edf1f5; text-align:left; font-size:13px; white-space:nowrap; }}
    th {{ color:#526173; background:#f7f9fb; font-weight:700; }}
    .note {{ color:var(--muted); font-size:13px; margin-top:10px; }}
    @media (max-width:900px) {{ .signal-grid,.card-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:28px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">信用债 AI 日内观察</div>
      <h1>{snapshot_dir.name} 盘中雷达：{overall}</h1>
      <p class="muted">数据来自 DM 合法接口：国债活跃券经纪商分钟线、实时最优报价、国债期货分钟线、基差与机构行为指数。本页是研究辅助，不构成交易指令。</p>
    </div>
  </header>
  <main class="wrap main">
    <section class="section">
      <h2>综合温度</h2>
      <div class="signal-grid">
        <div class="metric"><span>活跃国债平均日内变化</span><b>{_fmt_signed(avg_yield_change, 2, ' bp')}</b></div>
        <div class="metric"><span>国债期货平均涨跌</span><b>{_fmt_pct(avg_future_pct, 2)}</b></div>
        <div class="metric"><span>经纪商 bid-offer 均值</span><b>{_fmt_num(avg_quote_width, 2)} bp</b></div>
        <div class="metric"><span>机构指数均值</span><b>{_fmt_num(avg_sentiment, 1)}</b></div>
      </div>
      <p class="note">方向解释：收益率上行通常意味着利率债价格承压；国债期货上涨通常意味着利率债偏强。这里先做盘中温度计，不直接给信用债买卖指令。</p>
    </section>
    <section class="card-grid">{''.join(bond_cards)}</section>
    <section class="section">
      <h2>国债期货与基差</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>合约</th><th>对应期限</th><th>最新时间</th><th>最新价</th><th>日内价差</th><th>日内涨跌</th><th>基差</th><th>IRR</th><th>成交量</th></tr></thead>
          <tbody>{''.join(futures_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>实时最优报价</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期限</th><th>债券代码</th><th>报价时间</th><th>买收益率</th><th>卖收益率</th><th>bid-offer</th><th>成交较昨收</th><th>状态</th></tr></thead>
          <tbody>{''.join(quote_rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>机构行为指数</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>来源</th><th>指标</th><th>最新时间</th><th>均值</th><th>银行</th><th>券商</th><th>基金</th><th>状态</th></tr></thead>
          <tbody>{''.join(sentiment_rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
