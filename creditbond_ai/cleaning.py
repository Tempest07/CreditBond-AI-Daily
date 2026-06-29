from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    add_derived_features,
    coerce_numeric_frame,
    find_date_col,
    load_wide_dataset,
    parse_date_series,
    read_csv_auto,
    resample_and_fill,
)
from .dictionary import create_data_dictionary, validate_data_dictionary


def _missing_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        if col == "date":
            continue
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "列名": col,
                "缺失数": missing,
                "缺失率": float(df[col].isna().mean()) if len(df) else 0.0,
            }
        )
    return sorted(rows, key=lambda x: x["缺失率"], reverse=True)


def _numeric_coercion_issues(raw: pd.DataFrame, date_col: str) -> dict[str, int]:
    issues = {}
    parsed_dates = parse_date_series(raw[date_col])
    data_rows = raw.loc[parsed_dates.notna()].copy()
    for col in raw.columns:
        if col == date_col:
            continue
        original_non_empty = data_rows[col].notna() & (data_rows[col].astype(str).str.strip() != "")
        numeric = pd.to_numeric(data_rows[col], errors="coerce")
        bad_count = int((original_non_empty & numeric.isna()).sum())
        if bad_count:
            issues[str(col)] = bad_count
    return issues


def build_quality_report(
    raw_path: str | Path,
    raw_df: pd.DataFrame,
    loaded_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    feature_df: pd.DataFrame | None,
    date_col: str,
    freq: str,
    fill: str,
) -> dict:
    parsed_dates = parse_date_series(raw_df[date_col])
    valid_date_rows = raw_df.loc[parsed_dates.notna()].copy()
    valid_date_rows["_parsed_date"] = parsed_dates.loc[parsed_dates.notna()].values
    duplicate_dates = int(valid_date_rows["_parsed_date"].duplicated().sum())

    removed_all_missing_cols = [
        str(col)
        for col in raw_df.columns
        if col != date_col and str(col) not in [str(c) for c in loaded_df.columns]
    ]
    clean_start = cleaned_df["date"].min().strftime("%Y-%m-%d") if len(cleaned_df) else ""
    clean_end = cleaned_df["date"].max().strftime("%Y-%m-%d") if len(cleaned_df) else ""
    loaded_start = loaded_df["date"].min().strftime("%Y-%m-%d") if len(loaded_df) else ""

    return {
        "原始文件": str(raw_path),
        "日期列": date_col,
        "目标频率": freq,
        "填充方法": fill,
        "禁止向后填充": True,
        "原始形状": list(raw_df.shape),
        "有效日期行数": int(parsed_dates.notna().sum()),
        "无效日期行数": int(parsed_dates.isna().sum()),
        "重复日期数": duplicate_dates,
        "数值转换异常": _numeric_coercion_issues(raw_df, date_col),
        "全空或非数值列已移除": removed_all_missing_cols,
        "清洗前形状": list(loaded_df.shape),
        "清洗后形状": list(cleaned_df.shape),
        "增强特征形状": list(feature_df.shape) if feature_df is not None else None,
        "清洗后日期范围": [clean_start, clean_end],
        "因禁止向后填充而截掉的开头区间": [loaded_start, clean_start] if loaded_start != clean_start else [],
        "清洗前缺失概况": _missing_summary(loaded_df),
        "清洗后缺失概况": _missing_summary(cleaned_df),
        "备注": [
            "日频市场数据只允许向前填充。",
            "月度、季度、年度宏观指标后续必须按真实发布日期进入模型。",
            "如果同一日期有多条记录，默认保留最后一条。",
        ],
    }


def clean_dataset(
    input_path: str | Path,
    output_dir: str | Path = "data/cleaned",
    name: str | None = None,
    freq: str = "B",
    fill: str = "ffill",
    derive_features: bool = True,
    raw_metadata_path: str | Path | None = None,
) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = name or input_path.stem

    raw_df = read_csv_auto(input_path)
    date_col = find_date_col(raw_df.columns)
    loaded_df = load_wide_dataset(input_path)
    cleaned_df = resample_and_fill(loaded_df, freq=freq, fill=fill)
    feature_df = add_derived_features(cleaned_df) if derive_features else None

    cleaned_path = output_dir / f"{dataset_name}_清洗后.csv"
    dictionary_path = output_dir / f"{dataset_name}_数据字典.csv"
    report_path = output_dir / f"{dataset_name}_清洗报告.json"
    validation_path = output_dir / f"{dataset_name}_数据字典校验.json"
    feature_path = output_dir / f"{dataset_name}_增强特征.csv"
    feature_dictionary_path = output_dir / f"{dataset_name}_增强特征字典.csv"

    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    dictionary = create_data_dictionary(cleaned_path, dictionary_path, raw_path=raw_metadata_path or input_path)
    validation = validate_data_dictionary(cleaned_path, dictionary_path, report_path=validation_path)

    if feature_df is not None:
        feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
        create_data_dictionary(feature_path, feature_dictionary_path, raw_path=raw_metadata_path or input_path)

    report = build_quality_report(
        raw_path=input_path,
        raw_df=raw_df,
        loaded_df=loaded_df,
        cleaned_df=cleaned_df,
        feature_df=feature_df,
        date_col=date_col,
        freq=freq,
        fill=fill,
    )
    report["数据字典校验通过"] = bool(validation["是否通过"])
    report["输出文件"] = {
        "清洗后数据": str(cleaned_path),
        "数据字典": str(dictionary_path),
        "数据字典校验": str(validation_path),
        "清洗报告": str(report_path),
        "增强特征数据": str(feature_path) if feature_df is not None else "",
        "增强特征字典": str(feature_dictionary_path) if feature_df is not None else "",
    }
    report["指标数量"] = int(len(dictionary))
    report["增强后指标数量"] = int(feature_df.shape[1] - 1) if feature_df is not None else None

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def _rename_duplicate_columns(
    df: pd.DataFrame,
    existing_cols: set[str],
    source_stem: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    renamed = {}
    out = df.copy()
    for col in list(out.columns):
        if col == "date" or col not in existing_cols:
            continue
        new_col = f"{source_stem}_{col}"
        suffix = 2
        while new_col in existing_cols or new_col in out.columns:
            new_col = f"{source_stem}_{col}_{suffix}"
            suffix += 1
        renamed[col] = new_col
    if renamed:
        out = out.rename(columns=renamed)
    return out, renamed


def clean_data_folder(
    input_dir: str | Path,
    output_dir: str | Path = "data/cleaned_batch",
    pattern: str = "*.csv",
    name: str = "合并数据集",
    freq: str = "B",
    fill: str = "ffill",
    derive_features: bool = True,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    per_file_dir = output_dir / "per_file"
    combined_dir = output_dir / "combined"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files matched {pattern} in {input_dir}")

    per_file_reports = []
    cleaned_frames = []
    existing_cols: set[str] = {"date"}
    duplicate_renames: dict[str, dict[str, str]] = {}

    for file in files:
        report = clean_dataset(
            input_path=file,
            output_dir=per_file_dir,
            name=file.stem,
            freq=freq,
            fill=fill,
            derive_features=False,
            raw_metadata_path=file,
        )
        per_file_reports.append(report)
        cleaned_path = Path(report["输出文件"]["清洗后数据"])
        cleaned = load_wide_dataset(cleaned_path)
        cleaned, renamed = _rename_duplicate_columns(cleaned, existing_cols, file.stem)
        if renamed:
            duplicate_renames[file.name] = renamed
        existing_cols.update(c for c in cleaned.columns if c != "date")
        cleaned_frames.append(cleaned)

    combined = cleaned_frames[0]
    for frame in cleaned_frames[1:]:
        combined = pd.merge(combined, frame, on="date", how="outer")
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    combined_history = combined.copy()
    numeric_cols = [c for c in combined_history.columns if c != "date"]
    combined_history[numeric_cols] = combined_history[numeric_cols].ffill()
    combined_model_ready = combined_history.dropna(how="any").reset_index(drop=True)

    combined_history_path = combined_dir / f"{name}_历史合并.csv"
    combined_ready_path = combined_dir / f"{name}_模型就绪.csv"
    combined_dictionary_path = combined_dir / f"{name}_数据字典.csv"
    combined_validation_path = combined_dir / f"{name}_数据字典校验.json"
    combined_feature_path = combined_dir / f"{name}_增强特征.csv"
    combined_feature_dictionary_path = combined_dir / f"{name}_增强特征字典.csv"
    batch_report_path = combined_dir / f"{name}_批量清洗报告.json"

    combined_history.to_csv(combined_history_path, index=False, encoding="utf-8-sig")
    combined_model_ready.to_csv(combined_ready_path, index=False, encoding="utf-8-sig")
    create_data_dictionary(combined_ready_path, combined_dictionary_path)
    validation = validate_data_dictionary(combined_ready_path, combined_dictionary_path, report_path=combined_validation_path)

    feature_df = None
    if derive_features:
        feature_df = add_derived_features(combined_model_ready)
        feature_df.to_csv(combined_feature_path, index=False, encoding="utf-8-sig")
        create_data_dictionary(combined_feature_path, combined_feature_dictionary_path)

    history_start = combined_history["date"].min().strftime("%Y-%m-%d") if len(combined_history) else ""
    ready_start = combined_model_ready["date"].min().strftime("%Y-%m-%d") if len(combined_model_ready) else ""
    ready_end = combined_model_ready["date"].max().strftime("%Y-%m-%d") if len(combined_model_ready) else ""
    report = {
        "输入目录": str(input_dir),
        "匹配规则": pattern,
        "文件数量": len(files),
        "文件列表": [str(p) for p in files],
        "目标频率": freq,
        "填充方法": fill,
        "禁止向后填充": True,
        "单文件报告": [
            {
                "原始文件": item["原始文件"],
                "清洗后形状": item["清洗后形状"],
                "清洗后日期范围": item["清洗后日期范围"],
                "数据字典校验通过": item["数据字典校验通过"],
            }
            for item in per_file_reports
        ],
        "重复列重命名": duplicate_renames,
        "历史合并形状": list(combined_history.shape),
        "模型就绪形状": list(combined_model_ready.shape),
        "增强特征形状": list(feature_df.shape) if feature_df is not None else None,
        "历史合并开始日期": history_start,
        "模型就绪日期范围": [ready_start, ready_end],
        "因多源数据起点不同而截掉的开头区间": [history_start, ready_start] if history_start != ready_start else [],
        "数据字典校验通过": bool(validation["是否通过"]),
        "输出文件": {
            "单文件目录": str(per_file_dir),
            "历史合并数据": str(combined_history_path),
            "模型就绪数据": str(combined_ready_path),
            "合并数据字典": str(combined_dictionary_path),
            "合并数据字典校验": str(combined_validation_path),
            "增强特征数据": str(combined_feature_path) if feature_df is not None else "",
            "增强特征字典": str(combined_feature_dictionary_path) if feature_df is not None else "",
            "批量清洗报告": str(batch_report_path),
        },
    }
    with batch_report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report
