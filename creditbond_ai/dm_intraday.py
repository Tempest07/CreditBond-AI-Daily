from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .dm_api import _max_offset_from_result, _post_data_with_retry, _records_from_result, create_dm_client


BOND_BARS_PATH = "/dm-quant-func-service/api/v1/bond/market-data/bars"
BOND_DAILY_MARKET_PATH = "/dm-quant-func-service/api/v1/bond/market-data/date"
BOND_REALTIME_QUOTE_PATH = "/dm-quant-func-service/api/v1/bond/market-data/realtime-quote"
BOND_ROLLING_BONDS_PATH = "/dm-quant-func-service/api/v1/bond/market-data/rolling-bonds"
BOND_INSTI_SENTIMENT_PATH = "/dm-quant-func-service/api/v1/bond/analysis/insti-sentiment"
BOND_BASIC_INFO_PATH = "/dm-quant-func-service/api/v1/bond/basic-info/info"
BOND_OUTSTANDING_BONDS_PATH = "/dm-quant-func-service/api/v1/bond/basic-info/outstanding-bonds"
BOND_PRIMARY_PATH = "/dm-quant-func-service/api/v1/bond/primary/data"
MONEY_MARKET_DATE_PATH = "/dm-quant-func-service/api/v1/money-market/data/date"
MONEY_MARKET_SENTIMENT_PATH = "/dm-quant-func-service/api/v1/money-market/analysis/sentiment-index"
FUTURES_BARS_PATH = "/dm-quant-func-service/api/v1/futures/market-data/bars"
FUTURES_BASIS_PATH = "/dm-quant-func-service/api/v1/futures/analysis/basis"
FUTURES_VOL_OI_RANK_PATH = "/dm-quant-func-service/api/v1/futures/analysis/vol-oi-rank"


TBOND_ACTIVE_CODES = [
    "1YTBOND",
    "2YTBOND",
    "3YTBOND",
    "5YTBOND",
    "7YTBOND",
    "10YTBOND",
    "20YTBOND",
    "30YTBOND",
    "50YTBOND",
    "1YNDBOND",
    "2YNDBOND",
    "3YNDBOND",
    "5YNDBOND",
    "7YNDBOND",
    "10YNDBOND",
    "15YNDBOND",
    "20YNDBOND",
]

TREASURY_FUTURES_CODES = [
    "TS2609",
    "TF2609",
    "T2609",
    "TL2609",
]


def _list(values: Iterable[Any]) -> list[Any]:
    return [item for item in values if str(item).strip()]


def _frame_from_dm(client, path: str, payload: dict[str, Any]) -> pd.DataFrame:
    result = _post_data_with_retry(client, payload=payload, api_path=path)
    return pd.DataFrame(_records_from_result(result))


def fetch_bond_bars(
    security_ids: Iterable[str],
    start_datetime: str | date,
    end_datetime: str | date,
    kline_type: int = 2,
    data_sources: Iterable[int] = (1,),
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "securityIdList": _list(security_ids),
        "dataSourceList": _list(data_sources),
        "klineType": int(kline_type),
        "startDatetime": str(start_datetime),
        "endDatetime": str(end_datetime),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_BARS_PATH, payload)


def fetch_bond_daily_market(
    security_ids: Iterable[str],
    start_date: str | date,
    end_date: str | date,
    data_sources: Iterable[int] = (1,),
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "securityIdList": _list(security_ids),
        "dataSourceList": _list(data_sources),
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_DAILY_MARKET_PATH, payload)


def fetch_bond_realtime_quote(
    security_ids: Iterable[str],
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {"securityIdList": _list(security_ids)}
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_REALTIME_QUOTE_PATH, payload)


def fetch_bond_basic_info(
    security_ids: Iterable[str] | None = None,
    sec_short_names: Iterable[str] | None = None,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {}
    if security_ids:
        payload["securityIdList"] = _list(security_ids)
    if sec_short_names:
        payload["secShortNameList"] = _list(sec_short_names)
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_BASIC_INFO_PATH, payload)


def fetch_bond_outstanding_bonds(
    issuer_full_name: str | None = None,
    society_code: str | None = None,
    bond_status: Iterable[int] | None = None,
    field_names: list[str] | None = None,
    offset: str | int | None = None,
    client=None,
) -> tuple[pd.DataFrame, str | None]:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {}
    if issuer_full_name:
        payload["issuerFullName"] = issuer_full_name
    if society_code:
        payload["societyCode"] = society_code
    if bond_status:
        payload["bondStatusList"] = _list(bond_status)
    if field_names:
        payload["fieldNames"] = field_names
    if offset is not None and str(offset).strip():
        payload["offset"] = offset
    result = _post_data_with_retry(own_client, payload=payload, api_path=BOND_OUTSTANDING_BONDS_PATH)
    records = _records_from_result(result)
    return pd.DataFrame(records), _max_offset_from_result(result, records)


def fetch_bond_outstanding_bonds_all(
    issuer_full_name: str | None = None,
    society_code: str | None = None,
    bond_status: Iterable[int] | None = None,
    field_names: list[str] | None = None,
    max_pages: int = 20,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    frames: list[pd.DataFrame] = []
    offset: str | int | None = None
    for _ in range(max_pages):
        frame, next_offset = fetch_bond_outstanding_bonds(
            issuer_full_name=issuer_full_name,
            society_code=society_code,
            bond_status=bond_status,
            field_names=field_names,
            offset=offset,
            client=own_client,
        )
        frames.append(frame)
        if not next_offset or str(next_offset) == str(offset) or frame.empty:
            break
        offset = next_offset
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_bond_primary(
    start_date: str | date,
    end_date: str | date,
    bond_category: int | str | None = None,
    issuer_full_name: str | None = None,
    field_names: list[str] | None = None,
    offset: str | int | None = None,
    client=None,
) -> tuple[pd.DataFrame, str | None]:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if bond_category is not None and str(bond_category).strip():
        payload["bond_category"] = str(bond_category)
    if issuer_full_name:
        payload["issuerFullName"] = issuer_full_name
    if field_names:
        payload["fieldNames"] = field_names
    if offset is not None and str(offset).strip():
        payload["offset"] = offset
    result = _post_data_with_retry(own_client, payload=payload, api_path=BOND_PRIMARY_PATH)
    records = _records_from_result(result)
    return pd.DataFrame(records), _max_offset_from_result(result, records)


def fetch_bond_primary_all(
    start_date: str | date,
    end_date: str | date,
    bond_category: int | str | None = None,
    issuer_full_name: str | None = None,
    field_names: list[str] | None = None,
    max_pages: int = 50,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    frames: list[pd.DataFrame] = []
    offset: str | int | None = None
    for _ in range(max_pages):
        frame, next_offset = fetch_bond_primary(
            start_date=start_date,
            end_date=end_date,
            bond_category=bond_category,
            issuer_full_name=issuer_full_name,
            field_names=field_names,
            offset=offset,
            client=own_client,
        )
        frames.append(frame)
        if not next_offset or str(next_offset) == str(offset) or frame.empty:
            break
        offset = next_offset
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_rolling_bonds(
    key_tenors: Iterable[int],
    start_date: str | date,
    end_date: str | date,
    sequence_type: int = 1,
    bond_filter_type: int = 1,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "sequenceType": int(sequence_type),
        "bondFilterType": int(bond_filter_type),
        "keyTenor": _list(key_tenors),
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_ROLLING_BONDS_PATH, payload)


def extract_bond_codes(rolling_bonds: pd.DataFrame) -> list[str]:
    codes: list[str] = []
    for col in ("bondCode", "bond_code"):
        if col in rolling_bonds.columns:
            codes.extend(str(value) for value in rolling_bonds[col].dropna().tolist())
    return list(dict.fromkeys(code for code in codes if code.strip()))


def fetch_bond_insti_sentiment(
    data_source: int,
    start_datetime: str | date,
    end_datetime: str | date,
    freqs: Iterable[int] = (1,),
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "dataSource": int(data_source),
        "startDatetime": str(start_datetime),
        "endDatetime": str(end_datetime),
        "freqList": _list(freqs),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, BOND_INSTI_SENTIMENT_PATH, payload)


def fetch_futures_bars(
    security_ids: Iterable[str],
    start_datetime: str | date,
    end_datetime: str | date,
    kline_type: int = 2,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "securityIdList": _list(security_ids),
        "klineType": int(kline_type),
        "startDatetime": str(start_datetime),
        "endDatetime": str(end_datetime),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, FUTURES_BARS_PATH, payload)


def fetch_futures_basis(
    security_ids: Iterable[str],
    start_date: str | date,
    end_date: str | date,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "securityIdList": _list(security_ids),
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, FUTURES_BASIS_PATH, payload)


def fetch_money_market_date(
    start_date: str | date,
    end_date: str | date,
    instrument_types: Iterable[str] | None = None,
    instrument_codes: Iterable[str] | None = None,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if instrument_types:
        payload["instrumentTypeList"] = _list(instrument_types)
    if instrument_codes:
        payload["instrumentCodeList"] = _list(instrument_codes)
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, MONEY_MARKET_DATE_PATH, payload)


def fetch_money_market_sentiment(
    start_date: str | date,
    end_date: str | date,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, MONEY_MARKET_SENTIMENT_PATH, payload)


def fetch_futures_vol_oi_rank(
    security_ids: Iterable[str],
    start_date: str | date,
    end_date: str | date,
    field_names: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    own_client = client or create_dm_client(timeout=30)
    payload: dict[str, Any] = {
        "securityIdList": _list(security_ids),
        "startDate": str(start_date),
        "endDate": str(end_date),
    }
    if field_names:
        payload["fieldNames"] = field_names
    return _frame_from_dm(own_client, FUTURES_VOL_OI_RANK_PATH, payload)


def save_frame(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
