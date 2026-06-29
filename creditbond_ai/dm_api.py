from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WHEEL_PATH = Path("references/dm_api/dm_quant_api_client-0.2.3-py3-none-any.whl")
EDB_CODE_PATH = "/dm-quant-func-service/api/v1/edb/data-info/code"
EDB_DATA_PATH = "/dm-quant-func-service/api/v1/edb/data-info/data"
BOND_YIELD_CURVE_DATA_PATH = "/dm-quant-func-service/api/v1/bond/yield-curve/data"


def _ensure_dm_package(wheel_path: str | Path = DEFAULT_WHEEL_PATH):
    try:
        return importlib.import_module("dm_quant_api_client")
    except ModuleNotFoundError as exc:
        if exc.name != "dm_quant_api_client":
            raise
        wheel = Path(wheel_path)
        if wheel.exists():
            sys.path.insert(0, str(wheel.resolve()))
            try:
                return importlib.import_module("dm_quant_api_client")
            except ModuleNotFoundError as wheel_exc:
                if wheel_exc.name != "dm_quant_api_client":
                    raise
        raise RuntimeError(
            "未找到 dm_quant_api_client。请先安装 references/dm_api 下的 whl 包。"
        )


def create_dm_client(
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    wheel_path: str | Path = DEFAULT_WHEEL_PATH,
):
    try:
        module = _ensure_dm_package(wheel_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError("DM SDK 导入失败。") from exc
    try:
        client_cls = getattr(module, "DMQuantApiClient")
    except AttributeError as exc:
        raise RuntimeError("DM SDK 中未找到 DMQuantApiClient。") from exc
    try:
        return client_cls(
            app_key=app_key or os.getenv("INNO_APP_KEY"),
            app_secret=app_secret or os.getenv("INNO_APP_SECRET"),
            base_url=base_url,
            timeout=timeout,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "gmssl":
            raise RuntimeError("缺少 gmssl 依赖。请安装 gmssl 后再调用 DM 接口。") from exc
        raise


def _records_from_result(result) -> list[dict]:
    if isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("list", "records", "data", "rows"):
            value = result.get(key)
            if isinstance(value, list):
                return value
        return [result]
    return [{"value": result}]


def _max_offset_from_result(result, records: list[dict]) -> str | None:
    candidates = []
    if isinstance(result, dict):
        candidates.extend([
            result.get("Max_Offset"),
            result.get("max_Offset"),
            result.get("max_offset"),
            result.get("MaxOffset"),
            result.get("maxOffset"),
        ])
    if records:
        last = records[-1]
        candidates.extend([
            last.get("Max_Offset"),
            last.get("max_Offset"),
            last.get("max_offset"),
            last.get("MaxOffset"),
            last.get("maxOffset"),
        ])
    for item in candidates:
        if item is not None and str(item).strip():
            return str(item)
    return None


def _post_data_with_retry(client, payload: dict, api_path: str, retries: int = 4):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return client.post_data(data=payload, api_path=api_path, return_type="dict")
        except Exception as exc:
            last_error = exc
            text = str(exc)
            retryable = "429" in text or "请求过于频繁" in text
            if retryable and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("DM请求失败。")


def fetch_edb_indicator_codes(
    edb_level_ids: Iterable[str],
    field_names: list[str] | None = None,
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    max_pages: int = 100,
) -> pd.DataFrame:
    client = create_dm_client(app_key=app_key, app_secret=app_secret, base_url=base_url, timeout=timeout)
    all_records: list[dict] = []
    offset: str | None = None
    for _ in range(max_pages):
        payload = {"edbLevelIdList": list(edb_level_ids)}
        if field_names:
            payload["fieldNames"] = field_names
        if offset:
            payload["offset"] = offset
        result = _post_data_with_retry(client, payload=payload, api_path=EDB_CODE_PATH)
        records = _records_from_result(result)
        all_records.extend(records)
        next_offset = _max_offset_from_result(result, records)
        if not next_offset or next_offset == offset or not records:
            break
        offset = next_offset
    return pd.DataFrame(all_records)


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _chunk_days_for_frequency(frequency: str | None) -> int:
    text = str(frequency or "").lower()
    if any(key in text for key in ("日", "daily", "不定期")):
        return 360
    if any(key in text for key in ("周", "旬", "weekly")):
        return 365 * 4
    return 365 * 9


def iter_date_chunks(start_date: str, end_date: str, frequency: str | None = None):
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    step_days = _chunk_days_for_frequency(frequency)
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=step_days))
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def iter_month_chunks(start_date: str, end_date: str):
    """Yield chunks that respect DM yield-curve endpoint's 30-day limit."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=29))
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def fetch_edb_indicator_data(
    indicator_id: str,
    start_date: str,
    end_date: str,
    frequency: str | None = None,
    field_names: list[str] | None = None,
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    client = create_dm_client(app_key=app_key, app_secret=app_secret, base_url=base_url, timeout=timeout)
    frames = []
    for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, frequency=frequency):
        payload = {
            "indicatorId": indicator_id,
            "startDate": chunk_start,
            "endDate": chunk_end,
        }
        if field_names:
            payload["fieldNames"] = field_names
        result = _post_data_with_retry(client, payload=payload, api_path=EDB_DATA_PATH)
        records = _records_from_result(result)
        frames.append(pd.DataFrame(records))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "data_date" in out.columns:
        out = out.drop_duplicates(subset=["indicator_id", "data_date"], keep="last")
        out = out.sort_values(["indicator_id", "data_date"]).reset_index(drop=True)
    return out


def fetch_bond_yield_curve_data(
    data_source: int | str,
    curve_name: str,
    curve_terms: Iterable[str | int | float],
    start_date: str,
    end_date: str,
    curve_type: int | str = 1,
    field_names: list[str] | None = None,
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch DM bond yield-curve function data.

    DM limits this endpoint to at most 5 terms and a one-month date span per
    request, so this helper chunks date ranges while preserving one output
    table.
    """
    def _number_text(item) -> str:
        number = float(str(item).strip())
        return f"{number:g}"

    terms = [_number_text(item) for item in curve_terms if str(item).strip()]
    if not terms:
        raise ValueError("curve_terms 不能为空。")
    if len(terms) > 5:
        raise ValueError("DM收益率曲线函数单次调用 curveTermList 不能超过5个。")
    client = create_dm_client(app_key=app_key, app_secret=app_secret, base_url=base_url, timeout=timeout)
    frames = []
    for chunk_start, chunk_end in iter_month_chunks(start_date, end_date):
        payload = {
            "dataSource": int(float(data_source)),
            "curveName": curve_name,
            "curveTermList": terms,
            "curveType": int(float(curve_type)),
            "startDate": chunk_start,
            "endDate": chunk_end,
        }
        if field_names:
            payload["fieldNames"] = field_names
        result = _post_data_with_retry(client, payload=payload, api_path=BOND_YIELD_CURVE_DATA_PATH)
        records = _records_from_result(result)
        if records:
            frames.append(pd.DataFrame(records))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.rename(columns=_canonical_curve_column_map(out.columns))
    if "valuation_date" in out.columns and "curve_term" in out.columns:
        out["_valuation_date_parsed"] = parse_dm_china_datetime(out["valuation_date"])
        out["_curve_term_num"] = pd.to_numeric(out["curve_term"], errors="coerce")
        out = (
            out.sort_values(["_valuation_date_parsed", "_curve_term_num"])
            .drop_duplicates(subset=["_valuation_date_parsed", "_curve_term_num", "curve_ch_name"], keep="last")
            .drop(columns=["_valuation_date_parsed", "_curve_term_num"])
            .reset_index(drop=True)
        )
    return out


def _canonical_curve_column_map(columns: Iterable[str]) -> dict[str, str]:
    aliases = {
        "data_source": {"data_source", "dataSource"},
        "curve_ch_name": {"curve_ch_name", "curveChName"},
        "curve_term": {"curve_term", "curveTerm"},
        "curve_type": {"curve_type", "curveType"},
        "forward_n": {"forward_n", "forwardN"},
        "forward_k": {"forward_k", "forwardK"},
        "valuation_date": {"valuation_date", "valuationDate"},
        "yield": {"yield", "curveYield"},
    }
    mapping = {}
    for col in columns:
        for canonical, names in aliases.items():
            if str(col) in names:
                mapping[str(col)] = canonical
                break
    return mapping


def curve_data_to_edb_like_raw(
    curve_df: pd.DataFrame,
    indicator_id: str,
    alias: str,
    curve_term: str | int | float,
    frequency: str = "日",
    publish_lag_days: int = 1,
) -> pd.DataFrame:
    """Convert yield-curve function rows to the existing raw EDB-like schema."""
    if curve_df.empty:
        return pd.DataFrame()
    df = curve_df.rename(columns=_canonical_curve_column_map(curve_df.columns)).copy()
    if "curve_term" not in df.columns or "valuation_date" not in df.columns or "yield" not in df.columns:
        raise ValueError("收益率曲线函数返回缺少 curve_term / valuation_date / yield 字段。")
    wanted_term = float(curve_term)
    df["_curve_term_num"] = pd.to_numeric(df["curve_term"], errors="coerce")
    df = df[np.isclose(df["_curve_term_num"], wanted_term, equal_nan=False)].copy()
    if df.empty:
        return pd.DataFrame()
    valuation = parse_dm_china_datetime(df["valuation_date"]).dt.normalize()
    curve_name = df["curve_ch_name"] if "curve_ch_name" in df.columns else ""
    out = pd.DataFrame(
        {
            "indicator_id": str(indicator_id),
            "indicator_name": alias or (curve_name.astype(str) + f":{wanted_term:g}年"),
            "data_date": valuation.dt.strftime("%Y-%m-%d"),
            "data_value": pd.to_numeric(df["yield"], errors="coerce"),
            "basic_indicator_unit": "%",
            "publish_time": (valuation + pd.to_timedelta(publish_lag_days, unit="D")).dt.strftime("%Y-%m-%d"),
            "data_source": "DM_BOND_YIELD_CURVE_FUNC",
            "statistical_frequency": frequency or "日",
            "curve_ch_name": df["curve_ch_name"] if "curve_ch_name" in df.columns else "",
            "curve_term": df["_curve_term_num"],
            "curve_type": df["curve_type"] if "curve_type" in df.columns else "",
        }
    )
    out = out.dropna(subset=["data_date", "data_value"]).reset_index(drop=True)
    return out


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _canonical_column_map(columns: Iterable[str]) -> dict[str, str]:
    aliases = {
        "indicator_id": {"indicator_id", "indicatorId"},
        "indicator_name": {"indicator_name", "indicatorName"},
        "data_date": {"data_date", "dataDate"},
        "data_value": {"data_value", "dataValue"},
        "basic_indicator_unit": {"basic_indicator_unit", "basicIndicatorUnit"},
        "publish_time": {"publish_time", "publishTime"},
        "data_source": {"data_source", "dataSource"},
        "statistical_frequency": {"statistical_frequency", "statisticalFrequency"},
    }
    mapping = {}
    for col in columns:
        for canonical, names in aliases.items():
            if str(col) in names:
                mapping[str(col)] = canonical
                break
    return mapping


def parse_dm_datetime(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    ms_mask = numeric.notna() & (numeric.abs() >= 10**11)
    sec_mask = numeric.notna() & (numeric.abs() >= 10**8) & ~ms_mask
    if ms_mask.any():
        parsed.loc[ms_mask] = pd.to_datetime(numeric.loc[ms_mask], unit="ms", errors="coerce")
    if sec_mask.any():
        parsed.loc[sec_mask] = pd.to_datetime(numeric.loc[sec_mask], unit="s", errors="coerce")
    text_mask = parsed.isna()
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(values.loc[text_mask], errors="coerce")
    return parsed


def parse_dm_china_datetime(values: pd.Series) -> pd.Series:
    """Parse DM epoch timestamps as Asia/Shanghai wall dates.

    Some function endpoints encode a China-local midnight as epoch milliseconds.
    Parsing those timestamps as naive UTC shifts dates to the previous day, so
    curve valuation dates need this China-time parser.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    ms_mask = numeric.notna() & (numeric.abs() >= 10**11)
    sec_mask = numeric.notna() & (numeric.abs() >= 10**8) & ~ms_mask
    if ms_mask.any():
        parsed.loc[ms_mask] = (
            pd.to_datetime(numeric.loc[ms_mask], unit="ms", utc=True, errors="coerce")
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    if sec_mask.any():
        parsed.loc[sec_mask] = (
            pd.to_datetime(numeric.loc[sec_mask], unit="s", utc=True, errors="coerce")
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    text_mask = parsed.isna()
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(values.loc[text_mask], errors="coerce")
    return parsed


def read_edb_raw(input_path: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    input_path = Path(input_path)
    files = sorted(input_path.glob(pattern)) if input_path.is_dir() else [input_path]
    frames = []
    for file in files:
        if file.suffix.lower() == ".json":
            frame = pd.read_json(file)
        else:
            frame = pd.read_csv(file, encoding="utf-8-sig")
        frame["source_file"] = file.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"没有找到 EDB 原始数据文件: {input_path}")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns=_canonical_column_map(raw.columns))
    required = {"indicator_id", "data_date", "data_value"}
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"EDB原始数据缺少必要列: {missing}")
    return raw


def _safe_indicator_col(indicator_id: str, indicator_name: str | None, used: set[str]) -> str:
    base = str(indicator_name or indicator_id).strip()
    for bad in ["/", "\\", "*", "?", "\"", "<", ">", "|", "\n", "\r", "\t"]:
        base = base.replace(bad, "_")
    base = base or str(indicator_id)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def edb_raw_to_point_in_time_wide(
    input_path: str | Path,
    output_path: str | Path,
    dictionary_path: str | Path | None = None,
    model_ready_path: str | Path | None = None,
    pattern: str = "*.csv",
    freq: str = "B",
    start_date: str | None = None,
    end_date: str | None = None,
    fallback_release_lag_days: int = 10,
    model_ready_max_missing_ratio: float = 0.2,
) -> dict:
    raw = read_edb_raw(input_path, pattern=pattern)
    raw["data_date"] = parse_dm_datetime(raw["data_date"])
    raw["data_value"] = pd.to_numeric(raw["data_value"], errors="coerce")
    if "publish_time" in raw.columns:
        raw["publish_time"] = parse_dm_datetime(raw["publish_time"])
    else:
        raw["publish_time"] = pd.NaT
    fallback_publish = raw["data_date"] + pd.to_timedelta(fallback_release_lag_days, unit="D")
    raw["available_date"] = raw["publish_time"].fillna(fallback_publish).dt.normalize()
    raw = raw.dropna(subset=["indicator_id", "data_date", "data_value", "available_date"]).copy()
    if raw.empty:
        raise ValueError("EDB原始数据清洗后为空。")

    min_date = pd.Timestamp(start_date) if start_date else raw["available_date"].min()
    max_date = pd.Timestamp(end_date) if end_date else raw["available_date"].max()
    base = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq=freq)})

    wide = base.copy()
    dictionary_rows = []
    used_cols = {"date"}
    for indicator_id, part in raw.groupby("indicator_id", sort=False):
        part = part.sort_values(["available_date", "data_date"]).drop_duplicates("available_date", keep="last")
        indicator_name = part["indicator_name"].dropna().iloc[-1] if "indicator_name" in part.columns and part["indicator_name"].notna().any() else str(indicator_id)
        alias = part["alias"].dropna().iloc[-1] if "alias" in part.columns and part["alias"].notna().any() else ""
        output_col = _safe_indicator_col(str(indicator_id), str(alias or indicator_name), used_cols)
        sub = part[["available_date", "data_value"]].rename(columns={"available_date": "_available_date", "data_value": output_col})
        wide = pd.merge_asof(
            wide.sort_values("date"),
            sub.sort_values("_available_date"),
            left_on="date",
            right_on="_available_date",
            direction="backward",
        ).drop(columns=["_available_date"])
        dictionary_rows.append(
            {
                "列名": output_col,
                "指标名称": indicator_name,
                "本地别名": alias,
                "DM指标ID": indicator_id,
                "频率": part["statistical_frequency"].dropna().iloc[-1] if "statistical_frequency" in part.columns and part["statistical_frequency"].notna().any() else "",
                "单位": part["basic_indicator_unit"].dropna().iloc[-1] if "basic_indicator_unit" in part.columns and part["basic_indicator_unit"].notna().any() else "",
                "数据来源": part["data_source"].dropna().iloc[-1] if "data_source" in part.columns and part["data_source"].notna().any() else "",
                "可获得规则": "按publish_time进入模型；publish_time缺失时使用fallback_release_lag_days",
                "发布滞后天数": fallback_release_lag_days,
                "填充方法": "按公告日merge_asof后向前持有，不向后填充",
                "是否入模": "是",
                "是否目标候选": "否",
                "指标类别": "DM_EDB",
                "开始日期": part["data_date"].min().strftime("%Y-%m-%d"),
                "结束日期": part["data_date"].max().strftime("%Y-%m-%d"),
                "最早可用日": part["available_date"].min().strftime("%Y-%m-%d"),
                "最晚可用日": part["available_date"].max().strftime("%Y-%m-%d"),
                "缺失publish_time条数": int(part["publish_time"].isna().sum()),
                "备注": "",
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_ratio = wide.drop(columns=["date"]).isna().mean()
    kept_model_cols = ["date"] + missing_ratio[missing_ratio <= model_ready_max_missing_ratio].index.tolist()
    dropped_model_cols = missing_ratio[missing_ratio > model_ready_max_missing_ratio].index.tolist()
    model_ready_df = wide[kept_model_cols].dropna(how="any").reset_index(drop=True)
    model_ready_rows = int(model_ready_df.shape[0])
    if model_ready_path:
        model_ready_path = Path(model_ready_path)
        model_ready_path.parent.mkdir(parents=True, exist_ok=True)
        model_ready_df.to_csv(model_ready_path, index=False, encoding="utf-8-sig")

    dictionary_df = pd.DataFrame(dictionary_rows)
    if dictionary_path:
        dictionary_path = Path(dictionary_path)
        dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary_df.to_csv(dictionary_path, index=False, encoding="utf-8-sig")

    return {
        "输入行数": int(len(raw)),
        "指标数量": int(raw["indicator_id"].nunique()),
        "输出宽表": str(output_path),
        "输出形状": list(wide.shape),
        "模型就绪行数": model_ready_rows,
        "模型就绪列数": int(len(kept_model_cols)),
        "模型就绪最大缺失率": model_ready_max_missing_ratio,
        "模型就绪剔除列": dropped_model_cols,
        "模型就绪表": str(model_ready_path) if model_ready_path else "",
        "数据字典": str(dictionary_path) if dictionary_path else "",
        "日期范围": [wide["date"].min().strftime("%Y-%m-%d"), wide["date"].max().strftime("%Y-%m-%d")],
        "存在publish_time缺失": bool(raw["publish_time"].isna().any()),
    }


def fetch_edb_data_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> dict:
    config = pd.read_csv(config_path, encoding="utf-8-sig")
    if "indicator_id" not in config.columns:
        raise ValueError("配置文件必须包含 indicator_id 列。")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for _, row in config.iterrows():
        indicator_id = str(row["indicator_id"]).strip()
        if not indicator_id or indicator_id.lower() == "nan":
            continue
        row_start = start_date or str(row.get("start_date", "") or "").strip()
        row_end = end_date or str(row.get("end_date", "") or "").strip()
        if not row_start or not row_end:
            raise ValueError(f"{indicator_id} 缺少 start_date/end_date。")
        frequency = str(row.get("frequency", "") or row.get("statistical_frequency", "") or "").strip() or None
        data = fetch_edb_indicator_data(
            indicator_id=indicator_id,
            start_date=row_start,
            end_date=row_end,
            frequency=frequency,
            app_key=app_key,
            app_secret=app_secret,
            base_url=base_url,
            timeout=timeout,
        )
        if "alias" in config.columns and pd.notna(row.get("alias")):
            data["alias"] = str(row["alias"])
        output_file = output_dir / f"{indicator_id}.csv"
        data.to_csv(output_file, index=False, encoding="utf-8-sig")
        saved_files.append(str(output_file))
    return {"配置文件": str(config_path), "输出目录": str(output_dir), "文件数": len(saved_files), "文件列表": saved_files}


def credentials_help() -> str:
    return json.dumps(
        {
            "PowerShell临时设置": [
                "$env:INNO_APP_KEY='你的AppKey'",
                "$env:INNO_APP_SECRET='你的AppSecret'",
            ],
            "说明": "不要把密钥写进代码或提交到仓库。当前命令只从环境变量或命令行参数读取。",
        },
        ensure_ascii=False,
        indent=2,
    )


def check_dm_environment(wheel_path: str | Path = DEFAULT_WHEEL_PATH) -> dict:
    wheel = Path(wheel_path)
    result = {
        "wheel存在": wheel.exists(),
        "wheel路径": str(wheel),
        "requests可用": importlib.util.find_spec("requests") is not None,
        "pandas可用": importlib.util.find_spec("pandas") is not None,
        "gmssl可用": importlib.util.find_spec("gmssl") is not None,
        "INNO_APP_KEY已设置": bool(os.getenv("INNO_APP_KEY")),
        "INNO_APP_SECRET已设置": bool(os.getenv("INNO_APP_SECRET")),
        "dm_quant_api_client可导入": False,
        "提示": [],
    }
    try:
        _ensure_dm_package(wheel)
        result["dm_quant_api_client可导入"] = True
    except Exception as exc:
        result["提示"].append(f"SDK导入检查未通过: {exc}")
    if not result["gmssl可用"]:
        result["提示"].append("缺少 gmssl；真实调用前需要安装。")
    if not result["INNO_APP_KEY已设置"] or not result["INNO_APP_SECRET已设置"]:
        result["提示"].append("未检测到完整凭证环境变量。")
    return result
