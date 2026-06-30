from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .dm_api import _post_data_with_retry, _records_from_result, create_dm_client


BOND_BARS_PATH = "/dm-quant-func-service/api/v1/bond/market-data/bars"
BOND_REALTIME_QUOTE_PATH = "/dm-quant-func-service/api/v1/bond/market-data/realtime-quote"
BOND_ROLLING_BONDS_PATH = "/dm-quant-func-service/api/v1/bond/market-data/rolling-bonds"
BOND_INSTI_SENTIMENT_PATH = "/dm-quant-func-service/api/v1/bond/analysis/insti-sentiment"
FUTURES_BARS_PATH = "/dm-quant-func-service/api/v1/futures/market-data/bars"
FUTURES_BASIS_PATH = "/dm-quant-func-service/api/v1/futures/analysis/basis"


TBOND_ACTIVE_CODES = [
    "2YTBOND",
    "5YTBOND",
    "10YTBOND",
    "30YTBOND",
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


def save_frame(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
