from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dm_api import create_dm_client
from .dm_intraday import (
    TREASURY_FUTURES_CODES,
    fetch_bond_daily_market,
    fetch_bond_primary_all,
    fetch_bond_realtime_quote,
    fetch_futures_vol_oi_rank,
    fetch_money_market_date,
    fetch_money_market_sentiment,
    save_frame,
)


DEFAULT_MONEY_TYPES = ["DR", "R", "Shibor", "FDR", "FR", "GC", "R-"]
DEFAULT_KEY_MONEY_CODES = ["DR001", "DR007", "R001", "R007", "SHIBORON", "SHIBOR1W", "SHIBOR3M"]
DEFAULT_CREDIT_QUOTE_LIMIT = 80


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _num(value: Any) -> float:
    try:
        text = str(value).replace("*", "").strip()
        number = float(text)
    except (TypeError, ValueError):
        return math.nan
    return number if np.isfinite(number) else math.nan


def _series_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("*", "", regex=False), errors="coerce")


def _col(df: pd.DataFrame, *names: str) -> str | None:
    columns = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        hit = columns.get(name.lower())
        if hit:
            return hit
    return None


def _date_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        parsed = _parse_date_series(pd.Series([value])).iloc[0]
        if pd.isna(parsed):
            return str(value)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _time_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return text.split()[-1] if " " in text else text


def _chunked(items: list[Any], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _date_chunks(start: str | date, end: str | date, max_days: int):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    current = start_ts
    while current <= end_ts:
        chunk_end = min(end_ts, current + pd.Timedelta(days=max_days - 1))
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def _parse_date_series(values: pd.Series) -> pd.Series:
    raw = values.copy()
    text = raw.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    ms_mask = numeric.gt(100_000_000_000)
    if ms_mask.any():
        parsed.loc[ms_mask] = pd.to_datetime(numeric.loc[ms_mask], unit="ms", errors="coerce")

    ymd_mask = numeric.between(19000101, 21001231) & ~ms_mask
    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(text.loc[ymd_mask], format="%Y%m%d", errors="coerce")

    rest_mask = parsed.isna() & text.ne("") & text.str.lower().ne("nan")
    if rest_mask.any():
        parsed.loc[rest_mask] = pd.to_datetime(text.loc[rest_mask], errors="coerce")
    return parsed


def _percentile(history: Iterable[float], value: float) -> float:
    series = pd.Series(list(history), dtype="float64").dropna()
    if series.empty or not np.isfinite(value):
        return math.nan
    return float((series <= value).mean())


def _z_score(history: Iterable[float], value: float) -> float:
    series = pd.Series(list(history), dtype="float64").dropna()
    if len(series) < 3 or not np.isfinite(value):
        return math.nan
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((value - float(series.mean())) / std)


def _fmt_pct(value: Any, digits: int = 0) -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "-"
    return f"{number * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _fmt_signed(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "-"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.{digits}f}{suffix}"


def load_credit_watchlist(path: str | Path | None) -> list[str]:
    if not path:
        return []
    watchlist_path = Path(path)
    if not watchlist_path.exists():
        return []
    df = pd.read_csv(watchlist_path, encoding="utf-8-sig")
    for name in ("security_id", "securityId", "bond_code", "code"):
        if name in df.columns:
            return [str(value).strip() for value in df[name].dropna().tolist() if str(value).strip()]
    if len(df.columns):
        col = df.columns[0]
        return [str(value).strip() for value in df[col].dropna().tolist() if str(value).strip()]
    return []


def _latest_credit_codes(primary: pd.DataFrame, limit: int) -> list[str]:
    if primary.empty:
        return []
    code_col = _col(primary, "securityId", "security_id")
    if not code_col:
        return []
    work = primary.copy()
    date_col = _col(work, "issueStartDate", "issue_start_date", "subscribeDate", "subscribe_date")
    if date_col:
        work["_sort_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values("_sort_date", ascending=False)
    codes = [str(value).strip() for value in work[code_col].dropna().tolist() if str(value).strip()]
    return list(dict.fromkeys(codes))[:limit]


def fetch_market_context(
    as_of_date: str | date,
    out_dir: str | Path,
    history_days: int = 120,
    primary_lookback_days: int = 30,
    credit_quote_limit: int = DEFAULT_CREDIT_QUOTE_LIMIT,
    credit_watchlist: str | Path | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch slow and fast market context around the intraday snapshot.

    The output intentionally separates raw context from model-ready features:
    it can be used by the daily report, the intraday dashboard, or later
    training experiments without mixing concerns.
    """
    as_of = pd.to_datetime(as_of_date).date()
    history_start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(history_days))).date()
    primary_start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(primary_lookback_days))).date()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = create_dm_client(timeout=timeout)

    outputs: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    def _record(name: str, df: pd.DataFrame, path: Path, error: str = "") -> None:
        if df is not None and not df.empty:
            save_frame(df, path)
        elif df is not None:
            save_frame(df, path)
        outputs[name] = str(path)
        summary_rows.append({"name": name, "ok": not error, "rows": int(len(df)) if df is not None else 0, "error": error})

    primary_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(primary_start, as_of, 30):
        try:
            primary_frames.append(
                fetch_bond_primary_all(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    bond_category=1,
                    client=client,
                )
            )
        except Exception as exc:
            summary_rows.append({"name": f"bond_primary_credit_{chunk_start}_{chunk_end}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    primary = pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    _record("bond_primary_credit_history", primary, out / "bond_primary_credit_history.csv")

    money_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(history_start, as_of, 7):
        try:
            money_frames.append(
                fetch_money_market_date(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    instrument_types=DEFAULT_MONEY_TYPES,
                    client=client,
                )
            )
        except Exception as exc:
            summary_rows.append({"name": f"money_market_{chunk_start}_{chunk_end}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    money = pd.concat(money_frames, ignore_index=True) if money_frames else pd.DataFrame()
    _record("money_market_history", money, out / "money_market_history.csv")

    sentiment_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(history_start, as_of, 60):
        try:
            sentiment_frames.append(fetch_money_market_sentiment(start_date=chunk_start, end_date=chunk_end, client=client))
        except Exception as exc:
            summary_rows.append({"name": f"money_sentiment_{chunk_start}_{chunk_end}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    money_sentiment = pd.concat(sentiment_frames, ignore_index=True) if sentiment_frames else pd.DataFrame()
    _record("money_market_sentiment_history", money_sentiment, out / "money_market_sentiment_history.csv")

    rank_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(history_start, as_of, 7):
        try:
            rank_frames.append(
                fetch_futures_vol_oi_rank(
                    security_ids=TREASURY_FUTURES_CODES,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    client=client,
                )
            )
        except Exception as exc:
            summary_rows.append({"name": f"futures_rank_{chunk_start}_{chunk_end}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    futures_rank = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    _record("futures_vol_oi_rank_history", futures_rank, out / "futures_vol_oi_rank_history.csv")

    watchlist_codes = load_credit_watchlist(credit_watchlist)
    primary_codes = _latest_credit_codes(primary, credit_quote_limit)
    credit_codes = list(dict.fromkeys(watchlist_codes + primary_codes))[:credit_quote_limit]
    (out / "credit_quote_codes.json").write_text(json.dumps(credit_codes, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["credit_quote_codes"] = str(out / "credit_quote_codes.json")

    quote_frames: list[pd.DataFrame] = []
    for codes in _chunked(credit_codes, 20):
        try:
            quote_frames.append(fetch_bond_realtime_quote(codes, client=client))
        except Exception as exc:
            summary_rows.append({"name": f"credit_realtime_quote_{len(codes)}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
    _record("credit_realtime_quote", quotes, out / "credit_realtime_quote.csv")

    daily_frames: list[pd.DataFrame] = []
    daily_start = (pd.Timestamp(as_of) - pd.Timedelta(days=7)).date()
    for codes in _chunked(credit_codes[:40], 5):
        try:
            daily_frames.append(
                fetch_bond_daily_market(
                    codes,
                    start_date=daily_start,
                    end_date=as_of,
                    data_sources=(1, 2),
                    client=client,
                )
            )
        except Exception as exc:
            summary_rows.append({"name": f"credit_daily_market_{len(codes)}", "ok": False, "rows": 0, "error": str(exc)[:500]})
    credit_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    _record("credit_daily_market", credit_daily, out / "credit_daily_market.csv")

    summary = pd.DataFrame(summary_rows)
    save_frame(summary, out / "market_context_summary.csv")
    outputs["summary"] = str(out / "market_context_summary.csv")
    return {
        "as_of_date": as_of.isoformat(),
        "history_start": history_start.isoformat(),
        "history_days": int(history_days),
        "primary_lookback_days": int(primary_lookback_days),
        "credit_quote_limit": int(credit_quote_limit),
        "outputs": outputs,
        "summary": summary_rows,
    }


def _latest_money_by_code(money: pd.DataFrame) -> list[dict[str, Any]]:
    if money.empty:
        return []
    code_col = _col(money, "instrumentCode", "instrument_code")
    date_col = _col(money, "issueDate", "issue_date")
    close_col = _col(money, "closeYield", "close_yield", "waYield", "wa_yield")
    wa_col = _col(money, "waYield", "wa_yield", "closeYield", "close_yield")
    vol_col = _col(money, "tradeVolume", "trade_volume")
    if not code_col or not date_col or not close_col:
        return []
    work = money.copy()
    work["_date"] = _parse_date_series(work[date_col])
    work["_close"] = _series_num(work[close_col])
    work["_wa"] = _series_num(work[wa_col]) if wa_col else work["_close"]
    work["_volume"] = _series_num(work[vol_col]) if vol_col else np.nan
    rows = []
    for code, group in work.dropna(subset=["_date"]).groupby(code_col):
        group = group.sort_values("_date")
        latest = group.iloc[-1]
        close = _num(latest["_close"])
        history = group["_close"].dropna().tolist()
        rows.append(
            {
                "instrument_code": str(code),
                "latest_date": latest["_date"].strftime("%Y-%m-%d"),
                "close_yield": close,
                "wa_yield": _num(latest["_wa"]),
                "trade_volume": _num(latest["_volume"]),
                "percentile": _percentile(history, close),
                "z_score": _z_score(history, close),
            }
        )
    order = {code: idx for idx, code in enumerate(DEFAULT_KEY_MONEY_CODES)}
    rows = sorted(rows, key=lambda item: order.get(item["instrument_code"].upper(), 999))
    return rows


def _money_sentiment_summary(sentiment: pd.DataFrame) -> dict[str, Any]:
    if sentiment.empty:
        return {}
    date_col = _col(sentiment, "issueDate", "issue_date")
    time_col = _col(sentiment, "issueTime", "issue_time")
    value_col = _col(sentiment, "indexAll", "index_all")
    if not date_col or not value_col:
        return {}
    work = sentiment.copy()
    work["_date"] = _parse_date_series(work[date_col])
    work["_time"] = work[time_col].astype(str) if time_col else ""
    work["_value"] = _series_num(work[value_col])
    work = work.dropna(subset=["_date", "_value"]).sort_values(["_date", "_time"])
    if work.empty:
        return {}
    latest = work.iloc[-1]
    value = _num(latest["_value"])
    return {
        "latest_date": latest["_date"].strftime("%Y-%m-%d"),
        "latest_time": _time_text(latest["_time"]),
        "index_all": value,
        "percentile": _percentile(work["_value"].tolist(), value),
        "z_score": _z_score(work["_value"].tolist(), value),
        "index_sibs": _num(latest.get(_col(work, "indexSibs", "index_sibs") or "")),
        "index_smbs": _num(latest.get(_col(work, "indexSmbs", "index_smbs") or "")),
        "index_nbfis": _num(latest.get(_col(work, "indexNbfis", "index_nbfis") or "")),
    }


def _primary_supply_summary(primary: pd.DataFrame) -> dict[str, Any]:
    if primary.empty:
        return {}
    date_col = _col(primary, "issueStartDate", "issue_start_date", "subscribeDate", "subscribe_date")
    amount_col = _col(primary, "planIssueAmount", "plan_issue_amount", "actuIssueAmount", "actu_issue_amount")
    actual_col = _col(primary, "actuIssueAmount", "actu_issue_amount")
    mult_col = _col(primary, "compliantSubscriptionMult", "compliant_subscription_mult", "marginalMult", "marginal_mult")
    if not date_col:
        return {}
    work = primary.copy()
    work["_date"] = _parse_date_series(work[date_col])
    work["_plan_amount"] = _series_num(work[amount_col]) / 10000.0 if amount_col else 0.0
    work["_actual_amount"] = _series_num(work[actual_col]) / 10000.0 if actual_col else np.nan
    work["_mult"] = _series_num(work[mult_col]) if mult_col else np.nan
    daily = (
        work.dropna(subset=["_date"])
        .groupby("_date", as_index=False)
        .agg(issue_count=(date_col, "size"), plan_amount_bil=("_plan_amount", "sum"), actual_amount_bil=("_actual_amount", "sum"), avg_mult=("_mult", "mean"))
        .sort_values("_date")
    )
    if daily.empty:
        return {}
    latest = daily.iloc[-1]
    plan = _num(latest["plan_amount_bil"])
    count = _num(latest["issue_count"])
    return {
        "latest_date": latest["_date"].strftime("%Y-%m-%d"),
        "issue_count": int(count) if np.isfinite(count) else 0,
        "plan_amount_bil": plan,
        "actual_amount_bil": _num(latest["actual_amount_bil"]),
        "avg_subscription_mult": _num(latest["avg_mult"]),
        "count_percentile": _percentile(daily["issue_count"].tolist(), count),
        "amount_percentile": _percentile(daily["plan_amount_bil"].tolist(), plan),
        "daily": [
            {
                "date": row["_date"].strftime("%Y-%m-%d"),
                "issue_count": int(row["issue_count"]),
                "plan_amount_bil": _num(row["plan_amount_bil"]),
                "actual_amount_bil": _num(row["actual_amount_bil"]),
                "avg_subscription_mult": _num(row["avg_mult"]),
            }
            for _, row in daily.tail(15).iterrows()
        ],
    }


def _credit_quote_summary(quotes: pd.DataFrame) -> dict[str, Any]:
    if quotes.empty:
        return {}
    width_col = _col(quotes, "brokerBidYieldSubOfr", "broker_bid_yield_sub_ofr")
    trade_change_col = _col(quotes, "brokerTradeYieldSubYtdClose", "broker_trade_yield_sub_ytd_close")
    bid_col = _col(quotes, "brokerBidYield", "broker_bid_yield")
    ofr_col = _col(quotes, "brokerOfrYield", "broker_ofr_yield")
    time_col = _col(quotes, "brokerIssueTime", "broker_issue_time")
    sec_col = _col(quotes, "secShortName", "sec_short_name")
    code_col = _col(quotes, "securityId", "security_id")
    work = quotes.copy()
    if width_col:
        work["_width_bp"] = _series_num(work[width_col]).abs() * 100
    elif bid_col and ofr_col:
        work["_width_bp"] = (_series_num(work[bid_col]) - _series_num(work[ofr_col])).abs() * 100
    else:
        work["_width_bp"] = np.nan
    work["_trade_change_bp"] = _series_num(work[trade_change_col]) * 100 if trade_change_col else np.nan
    return {
        "quote_count": int(len(work)),
        "valid_width_count": int(work["_width_bp"].notna().sum()),
        "avg_bid_ofr_width_bp": _num(work["_width_bp"].mean()),
        "median_bid_ofr_width_bp": _num(work["_width_bp"].median()),
        "avg_trade_ytd_change_bp": _num(work["_trade_change_bp"].mean()),
        "latest_time": max([_time_text(v) for v in work[time_col].dropna().tolist()] or [""]) if time_col else "",
        "examples": [
            {
                "security_id": str(row.get(code_col, "")) if code_col else "",
                "sec_short_name": str(row.get(sec_col, "")) if sec_col else "",
                "broker_issue_time": _time_text(row.get(time_col, "")) if time_col else "",
                "broker_bid_yield": _num(row.get(bid_col, math.nan)) if bid_col else math.nan,
                "broker_ofr_yield": _num(row.get(ofr_col, math.nan)) if ofr_col else math.nan,
                "bid_ofr_width_bp": _num(row.get("_width_bp", math.nan)),
                "trade_ytd_change_bp": _num(row.get("_trade_change_bp", math.nan)),
            }
            for _, row in work.sort_values("_width_bp", ascending=False).head(12).iterrows()
        ],
    }


def _futures_rank_summary(rank: pd.DataFrame) -> dict[str, Any]:
    if rank.empty:
        return {}
    date_col = _col(rank, "issueDate", "issue_date")
    type_col = _col(rank, "rankingType", "ranking_type")
    sec_col = _col(rank, "securityId", "security_id")
    volume_col = _col(rank, "volume")
    change_col = _col(rank, "curVolumeDay", "cur_volume_day")
    if not date_col or not type_col or not sec_col or not volume_col:
        return {}
    work = rank.copy()
    work["_date"] = _parse_date_series(work[date_col])
    work["_volume"] = _series_num(work[volume_col])
    work["_change"] = _series_num(work[change_col]) if change_col else np.nan
    latest_date = work["_date"].max()
    latest = work[work["_date"].eq(latest_date)].copy()
    rows = []
    for sec, group in latest.groupby(sec_col):
        buy = group[group[type_col].astype(str).str.contains("买|long|buy", case=False, regex=True)]
        sell = group[group[type_col].astype(str).str.contains("卖|short|sell", case=False, regex=True)]
        trade = group[group[type_col].astype(str).str.contains("成交|volume|trade", case=False, regex=True)]
        rows.append(
            {
                "security_id": str(sec),
                "buy_top_volume": _num(buy["_volume"].sum()),
                "sell_top_volume": _num(sell["_volume"].sum()),
                "trade_top_volume": _num(trade["_volume"].sum()),
                "buy_top_change": _num(buy["_change"].sum()),
                "sell_top_change": _num(sell["_change"].sum()),
                "net_buy_volume": _num(buy["_volume"].sum() - sell["_volume"].sum()),
                "net_buy_change": _num(buy["_change"].sum() - sell["_change"].sum()),
            }
        )
    net_change = float(np.nansum([row["net_buy_change"] for row in rows]))
    return {
        "latest_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "",
        "net_buy_change": net_change,
        "rows": rows,
    }


def _intraday_snapshot_summary(snapshot_dir: Path) -> dict[str, Any]:
    feature_vector = _read_csv(snapshot_dir / "features" / "intraday_feature_vector.csv")
    if feature_vector.empty:
        return {}
    row = feature_vector.iloc[-1].to_dict()
    avg_yield = _num(row.get("tbond_avg_yield_change_bp"))
    avg_future = _num(row.get("future_avg_price_change_pct"))
    return {
        "snapshot_date": str(row.get("snapshot_date", snapshot_dir.name)),
        "tbond_avg_yield_change_bp": avg_yield,
        "future_avg_price_change_pct": avg_future,
        "quote_avg_bid_ofr_width_bp": _num(row.get("quote_avg_bid_ofr_width_bp")),
        "bond_sentiment_avg_latest_score": _num(row.get("sentiment_avg_latest_score")),
    }


def build_market_context_features(context_dir: str | Path, snapshot_dir: str | Path | None = None) -> dict[str, Any]:
    context = Path(context_dir)
    money = _read_csv(context / "money_market_history.csv")
    sentiment = _read_csv(context / "money_market_sentiment_history.csv")
    primary = _read_csv(context / "bond_primary_credit_history.csv")
    quotes = _read_csv(context / "credit_realtime_quote.csv")
    futures_rank = _read_csv(context / "futures_vol_oi_rank_history.csv")
    credit_daily = _read_csv(context / "credit_daily_market.csv")

    money_rows = _latest_money_by_code(money)
    money_by_code = {row["instrument_code"].upper(): row for row in money_rows}
    dr007 = money_by_code.get("DR007", {})
    r007 = money_by_code.get("R007", {})
    shibor_on = money_by_code.get("SHIBORON", {})
    money_sentiment = _money_sentiment_summary(sentiment)
    primary_supply = _primary_supply_summary(primary)
    credit_quote = _credit_quote_summary(quotes)
    futures_rank_summary = _futures_rank_summary(futures_rank)
    intraday = _intraday_snapshot_summary(Path(snapshot_dir)) if snapshot_dir else {}

    credit_daily_summary: dict[str, Any] = {}
    if not credit_daily.empty:
        change_col = _col(credit_daily, "yieldSubClose", "yield_sub_close", "ytmSubClose", "ytm_sub_close")
        date_col = _col(credit_daily, "issueDate", "issue_date")
        work = credit_daily.copy()
        if change_col:
            work["_yield_change_bp"] = _series_num(work[change_col])
            credit_daily_summary["avg_yield_change_bp"] = _num(work["_yield_change_bp"].mean())
        if date_col:
            credit_daily_summary["latest_date"] = _date_text(_parse_date_series(work[date_col]).max())
        credit_daily_summary["rows"] = int(len(work))

    pressure_parts = []
    diagnostics = []

    rate_pressure = 0.0
    if intraday:
        y_bp = _num(intraday.get("tbond_avg_yield_change_bp"))
        f_pct = _num(intraday.get("future_avg_price_change_pct"))
        if np.isfinite(y_bp):
            rate_pressure += np.clip(y_bp / 2.5, -1.5, 1.5) * 25
            diagnostics.append(f"国债活跃券日内平均变动 {_fmt_signed(y_bp, 2, ' bp')}")
        if np.isfinite(f_pct):
            rate_pressure += np.clip((-f_pct * 100) / 0.25, -1.5, 1.5) * 20
            diagnostics.append(f"国债期货平均涨跌 {_fmt_pct(f_pct, 2)}")
    pressure_parts.append(("利率盘中压力", rate_pressure))

    money_pressure = 0.0
    for name, item, weight in [("DR007", dr007, 16), ("R007", r007, 10), ("Shibor隔夜", shibor_on, 8)]:
        pct = _num(item.get("percentile")) if item else math.nan
        if np.isfinite(pct):
            money_pressure += (pct - 0.5) * 2 * weight
            diagnostics.append(f"{name} 历史分位 {_fmt_pct(pct)}")
    sent_pct = _num(money_sentiment.get("percentile")) if money_sentiment else math.nan
    if np.isfinite(sent_pct):
        money_pressure += (sent_pct - 0.5) * 2 * 12
        diagnostics.append(f"资金情绪紧张度分位 {_fmt_pct(sent_pct)}")
    pressure_parts.append(("资金面压力", money_pressure))

    supply_pressure = 0.0
    amount_pct = _num(primary_supply.get("amount_percentile")) if primary_supply else math.nan
    count_pct = _num(primary_supply.get("count_percentile")) if primary_supply else math.nan
    if np.isfinite(amount_pct):
        supply_pressure += (amount_pct - 0.5) * 2 * 10
        diagnostics.append(f"信用债一级计划发行规模分位 {_fmt_pct(amount_pct)}")
    if np.isfinite(count_pct):
        supply_pressure += (count_pct - 0.5) * 2 * 6
    pressure_parts.append(("一级供给压力", supply_pressure))

    quote_pressure = 0.0
    quote_width = _num(credit_quote.get("avg_bid_ofr_width_bp")) if credit_quote else math.nan
    quote_trade = _num(credit_quote.get("avg_trade_ytd_change_bp")) if credit_quote else math.nan
    if np.isfinite(quote_width):
        quote_pressure += np.clip((quote_width - 0.4) / 0.8, -1.0, 1.5) * 10
        diagnostics.append(f"信用债实时报价平均 bid-offer {_fmt_num(quote_width, 2)} bp")
    if np.isfinite(quote_trade):
        quote_pressure += np.clip(quote_trade / 2.0, -1.5, 1.5) * 10
        diagnostics.append(f"信用债最新成交较昨收 {_fmt_signed(quote_trade, 2, ' bp')}")
    pressure_parts.append(("信用债报价压力", quote_pressure))

    position_pressure = 0.0
    net_buy_change = _num(futures_rank_summary.get("net_buy_change")) if futures_rank_summary else math.nan
    if np.isfinite(net_buy_change):
        position_pressure += np.clip(-net_buy_change / 50000, -1.5, 1.5) * 8
        diagnostics.append(f"国债期货前排名席位净多变化 {_fmt_signed(net_buy_change, 0, ' 手')}")
    pressure_parts.append(("期货持仓压力", position_pressure))

    total_pressure = float(np.clip(np.nansum([value for _, value in pressure_parts]), -100, 100))
    if total_pressure >= 35:
        signal = "偏空"
        display_signal = "震荡偏空" if total_pressure < 55 else "看空"
        tone = "bearish"
        plain = "盘中利率、资金或供给压力偏高，当前更适合降低追涨冲动，关注信用债估值承压风险。"
    elif total_pressure <= -35:
        signal = "偏多"
        display_signal = "震荡偏多" if total_pressure > -55 else "看多"
        tone = "bullish"
        plain = "盘中压力指标整体缓和，当前对信用债价格相对友好，但仍应结合日报模型的期限信号。"
    else:
        signal = "震荡"
        display_signal = "震荡"
        tone = "range"
        plain = "盘中快变量没有形成足够强的一致方向，当前更适合观察，不宜把单一指标当成交易结论。"

    out = {
        "context_dir": str(context),
        "snapshot_dir": str(snapshot_dir or ""),
        "intraday": intraday,
        "money_market": money_rows,
        "money_sentiment": money_sentiment,
        "primary_supply": primary_supply,
        "credit_quote": credit_quote,
        "credit_daily": credit_daily_summary,
        "futures_position": futures_rank_summary,
        "pressure_parts": [{"name": name, "score": float(score)} for name, score in pressure_parts],
        "pressure_score": total_pressure,
        "signal": signal,
        "display_signal": display_signal,
        "tone": tone,
        "plain": plain,
        "diagnostics": diagnostics,
    }
    features_dir = context / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "market_context_features.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    flat_rows = []
    for name, value in pressure_parts:
        flat_rows.append({"metric": name, "value": float(value)})
    flat_rows.append({"metric": "总压力分", "value": total_pressure})
    save_frame(pd.DataFrame(flat_rows), features_dir / "market_context_scores.csv")
    return out


def _bar(score: float) -> str:
    score = max(-100, min(100, _num(score)))
    width = abs(score)
    cls = "bad" if score > 0 else "good"
    return f'<div class="bar"><i class="{cls}" style="width:{width:.1f}%"></i></div>'


def render_realtime_dashboard(
    context_dir: str | Path,
    snapshot_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    refresh_seconds: int = 120,
) -> Path:
    context = Path(context_dir)
    data = build_market_context_features(context, snapshot_dir=snapshot_dir)
    out = Path(out_path) if out_path else context / "reports" / "intraday_market_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tone = data.get("tone", "range")
    score = _num(data.get("pressure_score"))
    signal = str(data.get("display_signal", "震荡"))

    pressure_rows = "\n".join(
        f"""
        <div class="score-row">
          <div><strong>{item['name']}</strong><span>{_fmt_signed(item['score'], 1)}</span></div>
          {_bar(item['score'])}
        </div>
        """
        for item in data.get("pressure_parts", [])
    )
    money_rows = "\n".join(
        f"""
        <tr><td>{row.get('instrument_code')}</td><td>{row.get('latest_date')}</td><td>{_fmt_num(row.get('close_yield'), 3)}%</td><td>{_fmt_pct(row.get('percentile'))}</td><td>{_fmt_signed(row.get('z_score'), 2)}</td></tr>
        """
        for row in data.get("money_market", [])[:12]
    )
    quote_rows = "\n".join(
        f"""
        <tr><td>{item.get('security_id')}</td><td>{item.get('sec_short_name')}</td><td>{item.get('broker_issue_time')}</td><td>{_fmt_num(item.get('broker_bid_yield'), 4)}%</td><td>{_fmt_num(item.get('broker_ofr_yield'), 4)}%</td><td>{_fmt_num(item.get('bid_ofr_width_bp'), 2)} bp</td><td>{_fmt_signed(item.get('trade_ytd_change_bp'), 2, ' bp')}</td></tr>
        """
        for item in (data.get("credit_quote") or {}).get("examples", [])
    )
    supply_rows = "\n".join(
        f"""
        <tr><td>{row.get('date')}</td><td>{row.get('issue_count')}</td><td>{_fmt_num(row.get('plan_amount_bil'), 1)} 亿元</td><td>{_fmt_num(row.get('actual_amount_bil'), 1)} 亿元</td><td>{_fmt_num(row.get('avg_subscription_mult'), 2)}</td></tr>
        """
        for row in (data.get("primary_supply") or {}).get("daily", [])
    )
    futures_rows = "\n".join(
        f"""
        <tr><td>{row.get('security_id')}</td><td>{_fmt_num(row.get('buy_top_volume'), 0)}</td><td>{_fmt_num(row.get('sell_top_volume'), 0)}</td><td>{_fmt_signed(row.get('net_buy_volume'), 0)}</td><td>{_fmt_signed(row.get('net_buy_change'), 0)}</td></tr>
        """
        for row in (data.get("futures_position") or {}).get("rows", [])
    )
    diagnostics = "".join(f"<li>{item}</li>" for item in data.get("diagnostics", []))
    sent = data.get("money_sentiment") or {}
    supply = data.get("primary_supply") or {}
    quote = data.get("credit_quote") or {}
    intraday = data.get("intraday") or {}

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{int(refresh_seconds)}">
  <title>信用债 AI 日内市场情绪</title>
  <style>
    :root {{
      --ink:#16202d; --muted:#637083; --line:#dde6ef; --bg:#edf3f7; --paper:#fff;
      --red:#cf3f3f; --green:#25825f; --gold:#b57b22; --blue:#2d6f9f;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--ink); }}
    .wrap {{ width:min(1280px, calc(100% - 36px)); margin:0 auto; }}
    .top {{ background:linear-gradient(180deg,#ffffff 0%,#f7fafc 100%); border-bottom:1px solid var(--line); }}
    .hero {{ display:grid; grid-template-columns:1fr 300px; gap:22px; padding:28px 0 24px; align-items:center; }}
    .eyebrow {{ color:var(--blue); font-weight:800; font-size:14px; margin-bottom:8px; }}
    h1 {{ margin:0; font-size:34px; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:21px; }}
    p {{ line-height:1.7; margin:12px 0 0; color:var(--muted); }}
    .signal {{ border:1px solid var(--line); border-radius:8px; background:#fff; padding:18px; }}
    .signal span {{ display:block; color:var(--muted); font-size:13px; }}
    .signal strong {{ display:block; margin:8px 0; font-size:34px; }}
    .signal strong.bullish {{ color:var(--red); }}
    .signal strong.bearish {{ color:var(--green); }}
    .signal strong.range {{ color:var(--gold); }}
    .main {{ display:grid; gap:16px; padding:18px 0 44px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .section,.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 20px rgba(20,35,55,.05); }}
    .section {{ padding:18px; }}
    .card {{ padding:15px; }}
    .card span {{ display:block; color:var(--muted); font-size:12px; }}
    .card b {{ display:block; margin-top:7px; font-size:20px; }}
    .score-row {{ display:grid; grid-template-columns:230px 1fr; gap:14px; align-items:center; margin:10px 0; }}
    .score-row span {{ color:var(--muted); margin-left:8px; }}
    .bar {{ height:12px; background:#e9eef3; border-radius:999px; overflow:hidden; }}
    .bar i {{ display:block; height:100%; border-radius:999px; }}
    .bar i.bad {{ background:var(--green); }}
    .bar i.good {{ background:var(--red); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; min-width:760px; border-collapse:collapse; background:#fff; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid #edf1f5; text-align:left; white-space:nowrap; font-size:13px; }}
    th {{ background:#f7f9fb; color:#536174; }}
    ul {{ margin:0; padding-left:20px; color:var(--muted); line-height:1.8; }}
    .foot {{ color:var(--muted); font-size:12px; }}
    @media (max-width:900px) {{ .hero,.grid {{ grid-template-columns:1fr; }} .score-row {{ grid-template-columns:1fr; }} h1 {{ font-size:28px; }} }}
  </style>
</head>
<body>
  <header class="top">
    <div class="wrap hero">
      <div>
        <div class="eyebrow">信用债 AI 日内市场情绪</div>
        <h1>当前信号：{signal}</h1>
        <p>{data.get('plain')}</p>
        <p class="foot">生成时间：{now_text}；页面每 {int(refresh_seconds)} 秒自动刷新一次。刷新页面只更新展示，重新抓取数据需要运行实时脚本。</p>
      </div>
      <div class="signal">
        <span>市场压力分</span>
        <strong class="{tone}">{_fmt_signed(score, 1)}</strong>
        <span>正数偏压力，负数偏友好</span>
      </div>
    </div>
  </header>
  <main class="wrap main">
    <section class="grid">
      <div class="card"><span>国债活跃券日内</span><b>{_fmt_signed(intraday.get('tbond_avg_yield_change_bp'), 2, ' bp')}</b></div>
      <div class="card"><span>国债期货日内</span><b>{_fmt_pct(intraday.get('future_avg_price_change_pct'), 2)}</b></div>
      <div class="card"><span>资金情绪指数</span><b>{_fmt_num(sent.get('index_all'), 1)}</b></div>
      <div class="card"><span>信用债报价样本</span><b>{quote.get('quote_count', 0)} 条</b></div>
      <div class="card"><span>DR007 历史分位</span><b>{_fmt_pct(next((r.get('percentile') for r in data.get('money_market', []) if str(r.get('instrument_code')).upper() == 'DR007'), math.nan))}</b></div>
      <div class="card"><span>资金情绪分位</span><b>{_fmt_pct(sent.get('percentile'))}</b></div>
      <div class="card"><span>一级发行规模分位</span><b>{_fmt_pct(supply.get('amount_percentile'))}</b></div>
      <div class="card"><span>信用债平均报价宽度</span><b>{_fmt_num(quote.get('avg_bid_ofr_width_bp'), 2)} bp</b></div>
    </section>

    <section class="section">
      <h2>压力拆解</h2>
      {pressure_rows}
      <ul>{diagnostics}</ul>
    </section>

    <section class="section">
      <h2>资金面：当前读数与历史分位</h2>
      <div class="table-wrap">
        <table><thead><tr><th>品种</th><th>日期</th><th>收盘/加权利率</th><th>历史分位</th><th>Z 分数</th></tr></thead><tbody>{money_rows}</tbody></table>
      </div>
    </section>

    <section class="section">
      <h2>信用债实时经纪商报价样本</h2>
      <div class="table-wrap">
        <table><thead><tr><th>代码</th><th>简称</th><th>报价时间</th><th>Bid</th><th>Offer</th><th>Bid-Offer</th><th>成交较昨收</th></tr></thead><tbody>{quote_rows}</tbody></table>
      </div>
    </section>

    <section class="section">
      <h2>一级发行：近日报供给</h2>
      <div class="table-wrap">
        <table><thead><tr><th>日期</th><th>数量</th><th>计划发行</th><th>实际发行</th><th>平均认购倍数</th></tr></thead><tbody>{supply_rows}</tbody></table>
      </div>
    </section>

    <section class="section">
      <h2>国债期货盘后持仓排名</h2>
      <div class="table-wrap">
        <table><thead><tr><th>合约</th><th>前排名多单</th><th>前排名空单</th><th>净多持仓</th><th>净多变化</th></tr></thead><tbody>{futures_rows}</tbody></table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out
