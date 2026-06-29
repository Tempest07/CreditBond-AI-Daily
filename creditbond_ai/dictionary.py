from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import find_date_col, load_wide_dataset, parse_date_series, read_csv_auto


DICTIONARY_COLUMNS = [
    "列名",
    "指标名称",
    "Wind代码",
    "频率",
    "单位",
    "数据来源",
    "可获得规则",
    "发布滞后天数",
    "填充方法",
    "是否入模",
    "是否目标候选",
    "指标类别",
    "开始日期",
    "结束日期",
    "缺失数",
    "缺失率",
    "备注",
]

REQUIRED_COLUMNS = [
    "列名",
    "指标名称",
    "频率",
    "可获得规则",
    "发布滞后天数",
    "填充方法",
    "是否入模",
]


def _first_value(df: pd.DataFrame, row_name: str, col: str) -> str:
    row = df[df.iloc[:, 0].astype(str).str.strip() == row_name]
    if row.empty or col not in row.columns:
        return ""
    value = row.iloc[0][col]
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_wind_metadata(raw_path: str | Path | None) -> dict[str, dict[str, str]]:
    if raw_path is None:
        return {}
    raw = read_csv_auto(raw_path)
    metadata: dict[str, dict[str, str]] = {}
    for col in raw.columns[1:]:
        metadata[str(col)] = {
            "Wind代码": _first_value(raw, "指标ID", col),
            "频率": _first_value(raw, "频率", col),
            "单位": _first_value(raw, "单位", col),
            "数据来源": _first_value(raw, "来源", col),
        }
    return metadata


def infer_frequency(dates: pd.Series) -> str:
    parsed = parse_date_series(dates).dropna().sort_values()
    if len(parsed) < 3:
        return ""
    median_days = parsed.diff().dropna().dt.days.median()
    if median_days <= 3:
        return "日"
    if median_days <= 10:
        return "周"
    if median_days <= 45:
        return "月"
    if median_days <= 120:
        return "季"
    return "年"


def guess_category(col: str) -> str:
    if "信用利差" in col:
        return "信用利差"
    if "期限利差" in col:
        return "期限利差"
    if "波动" in col:
        return "历史波动"
    if "变动" in col:
        return "历史变化"
    if "国债" in col and "收益率" in col:
        return "国债收益率"
    if "中短期票据" in col or "信用债" in col:
        return "信用债收益率"
    if any(key in col.lower() for key in ["pmi", "cpi", "ppi", "m2", "gdp"]):
        return "宏观指标"
    return "其他"


def guess_fill_method(freq: str) -> str:
    if freq in {"日", "周"}:
        return "向前填充"
    if freq in {"月", "季", "年"}:
        return "按发布日期滞后后向前填充"
    return "待确认"


def guess_availability_rule(freq: str, category: str) -> str:
    if freq == "日":
        if "收益率" in category or "利差" in category:
            return "收盘后可得；若盘中决策需滞后1日"
        return "当日或下一交易日可得，需按数据源确认"
    if freq in {"月", "季", "年"}:
        return "按真实发布日期进入模型，不按所属期提前使用"
    return "待确认"


def guess_release_lag(freq: str) -> str:
    if freq == "日":
        return "0"
    if freq == "周":
        return "1"
    if freq == "月":
        return "10"
    if freq == "季":
        return "30"
    return ""


def create_data_dictionary(
    data_path: str | Path,
    output_path: str | Path,
    raw_path: str | Path | None = None,
) -> pd.DataFrame:
    data = load_wide_dataset(data_path)
    metadata = read_wind_metadata(raw_path)
    inferred_freq = infer_frequency(data["date"])
    rows = []
    for col in data.columns:
        if col == "date":
            continue
        series = pd.to_numeric(data[col], errors="coerce")
        meta = metadata.get(col, {})
        freq = meta.get("频率") or inferred_freq
        category = guess_category(col)
        rows.append(
            {
                "列名": col,
                "指标名称": col,
                "Wind代码": meta.get("Wind代码", ""),
                "频率": freq,
                "单位": meta.get("单位", ""),
                "数据来源": meta.get("数据来源", ""),
                "可获得规则": guess_availability_rule(freq, category),
                "发布滞后天数": guess_release_lag(freq),
                "填充方法": guess_fill_method(freq),
                "是否入模": "是",
                "是否目标候选": "是" if "信用债收益率" in category else "否",
                "指标类别": category,
                "开始日期": data.loc[series.notna(), "date"].min().strftime("%Y-%m-%d") if series.notna().any() else "",
                "结束日期": data.loc[series.notna(), "date"].max().strftime("%Y-%m-%d") if series.notna().any() else "",
                "缺失数": int(series.isna().sum()),
                "缺失率": float(series.isna().mean()),
                "备注": "",
            }
        )
    out = pd.DataFrame(rows, columns=DICTIONARY_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out


def validate_data_dictionary(
    data_path: str | Path,
    dictionary_path: str | Path,
    report_path: str | Path | None = None,
) -> dict:
    data = load_wide_dataset(data_path)
    dictionary = read_csv_auto(dictionary_path)
    data_cols = [c for c in data.columns if c != "date"]
    dict_cols = dictionary["列名"].astype(str).tolist() if "列名" in dictionary.columns else []

    missing_in_dictionary = [c for c in data_cols if c not in dict_cols]
    extra_in_dictionary = [c for c in dict_cols if c not in data_cols]
    missing_required_fields = {}
    for _, row in dictionary.iterrows():
        name = str(row.get("列名", "")).strip()
        if not name:
            continue
        empty = []
        for field in REQUIRED_COLUMNS:
            value = row.get(field, "")
            if pd.isna(value) or str(value).strip() == "":
                empty.append(field)
        if empty:
            missing_required_fields[name] = empty

    lag_warnings = []
    for _, row in dictionary.iterrows():
        freq = str(row.get("频率", "")).strip()
        lag = row.get("发布滞后天数", "")
        availability = str(row.get("可获得规则", "")).strip()
        if freq in {"月", "季", "年"} and (pd.isna(lag) or str(lag).strip() in {"", "0"}):
            lag_warnings.append(f"{row.get('列名', '')}: 低频指标需要确认真实发布日期或滞后天数")
        if "待确认" in availability:
            lag_warnings.append(f"{row.get('列名', '')}: 可获得规则仍为待确认")

    report = {
        "数据文件": str(data_path),
        "数据字典": str(dictionary_path),
        "数据列数": len(data_cols),
        "字典行数": len(dictionary),
        "字典缺失列": missing_in_dictionary,
        "字典多余列": extra_in_dictionary,
        "必填字段缺失": missing_required_fields,
        "时点规则提示": lag_warnings,
        "是否通过": not missing_in_dictionary and not missing_required_fields,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return report
