from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATE_CANDIDATES = {
    "date",
    "datetime",
    "time",
    "trade_dt",
    "tradedate",
    "交易日期",
    "日期",
    "时间",
    "指标名称",
}


@dataclass
class WindowedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dates_train: np.ndarray
    dates_val: np.ndarray
    dates_test: np.ndarray
    changes_train: np.ndarray
    changes_val: np.ndarray
    changes_test: np.ndarray
    feature_cols: list[str]
    scaler: StandardScaler
    theta: float
    target_col: str
    horizon: int
    window: int


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    """Read CSV files from DM processing outputs or legacy sheets."""
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def find_date_col(columns: Iterable[str]) -> str:
    for col in columns:
        normalized = str(col).strip().lower().replace(" ", "").replace("_", "")
        if normalized in {c.replace("_", "") for c in DATE_CANDIDATES}:
            return str(col)
    for col in columns:
        text = str(col).strip().lower()
        if "date" in text or "日期" in text:
            return str(col)
    raise ValueError("No date column found. Please include a date column.")


def sanitize_column_name(name: str) -> str:
    name = Path(str(name)).stem
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def parse_date_series(values: pd.Series) -> pd.Series:
    """Parse common DM, legacy vendor, and spreadsheet date formats."""
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y/%m/%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="%Y%m%d", errors="coerce")
    return parsed


def coerce_numeric_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == date_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    numeric_cols = [c for c in out.columns if c == date_col or not out[c].isna().all()]
    return out[numeric_cols]


def _safe_feature_name(name: str) -> str:
    return (
        str(name)
        .replace(":", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-only yield-curve features without leaking future information."""
    out = df.copy()
    numeric_cols = [c for c in out.columns if c != "date" and pd.api.types.is_numeric_dtype(out[c])]

    for col in numeric_cols:
        base = _safe_feature_name(col)
        out[f"{base}_变动_1日"] = out[col].diff(1)
        out[f"{base}_变动_5日"] = out[col].diff(5)
        out[f"{base}_变动_20日"] = out[col].diff(20)
        out[f"{base}_波动_20日"] = out[col].diff(1).rolling(20, min_periods=5).std()

    tenors = ["1年", "3年", "5年", "7年", "10年", "20年"]
    treasury_by_tenor = {tenor: next((c for c in numeric_cols if "国债" in c and tenor in c), None) for tenor in tenors}
    credit_by_tenor = {
        tenor: next((c for c in numeric_cols if "中短期票据" in c and "AAA" in c and tenor in c), None)
        for tenor in tenors
    }

    for tenor in tenors:
        treasury_col = treasury_by_tenor.get(tenor)
        credit_col = credit_by_tenor.get(tenor)
        if treasury_col and credit_col:
            out[f"AAA信用利差_{tenor}"] = out[credit_col] - out[treasury_col]

    for label, cols in [("国债", treasury_by_tenor), ("AAA", credit_by_tenor)]:
        one = cols.get("1年")
        three = cols.get("3年")
        five = cols.get("5年")
        ten = cols.get("10年")
        twenty = cols.get("20年")
        if one and three:
            out[f"{label}期限利差_3年减1年"] = out[three] - out[one]
        if one and five:
            out[f"{label}期限利差_5年减1年"] = out[five] - out[one]
        if one and ten:
            out[f"{label}期限利差_10年减1年"] = out[ten] - out[one]
        if one and twenty:
            out[f"{label}期限利差_20年减1年"] = out[twenty] - out[one]

    return out.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")


def load_wide_dataset(path: str | Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    date_col = find_date_col(df.columns)
    df = df.rename(columns={date_col: "date"})
    df["date"] = parse_date_series(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df = coerce_numeric_frame(df, "date")
    value_cols = [c for c in df.columns if c != "date"]
    if not value_cols:
        raise ValueError("No numeric factor columns found in the dataset.")
    return df


def prepare_wide_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
    freq: str = "B",
    fill: str = "ffill",
    derive_features: bool = False,
) -> pd.DataFrame:
    df = load_wide_dataset(input_path)
    prepared = resample_and_fill(df, freq=freq, fill=fill)
    if derive_features:
        prepared = add_derived_features(prepared)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(out, index=False, encoding="utf-8-sig")
    return prepared


def resample_and_fill(df: pd.DataFrame, freq: str = "B", fill: str = "ffill") -> pd.DataFrame:
    work = df.copy()
    work["date"] = parse_date_series(work["date"])
    work = work.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    work = work.set_index("date").resample(freq).last()
    if fill == "ffill":
        work = work.ffill().dropna(how="any")
    elif fill == "interpolate":
        work = work.interpolate().ffill().dropna(how="any")
    elif fill == "none":
        pass
    else:
        raise ValueError("fill must be one of: ffill, interpolate, none")
    return work.reset_index()


def load_mapping(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    with mapping_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def infer_value_column(df: pd.DataFrame, date_col: str) -> str:
    numeric_scores: list[tuple[int, str]] = []
    for col in df.columns:
        if col == date_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_scores.append((int(numeric.notna().sum()), str(col)))
    numeric_scores.sort(reverse=True)
    if not numeric_scores or numeric_scores[0][0] == 0:
        raise ValueError("No numeric value column found.")
    return numeric_scores[0][1]


def merge_exported_files(
    input_dir: str | Path,
    output_path: str | Path,
    mapping_path: str | Path | None = None,
    freq: str = "B",
    fill: str = "ffill",
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    mapping = load_mapping(mapping_path)
    frames = []
    for file in files:
        df = read_csv_auto(file)
        date_col = find_date_col(df.columns)
        value_col = infer_value_column(df, date_col)
        out_col = mapping.get(file.name) or mapping.get(file.stem) or sanitize_column_name(file.stem)
        one = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: out_col})
        one["date"] = parse_date_series(one["date"])
        one[out_col] = pd.to_numeric(one[out_col], errors="coerce")
        frames.append(one.dropna(subset=["date"]).sort_values("date"))

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="outer")
    merged = merged.sort_values("date").drop_duplicates("date", keep="last")
    prepared = resample_and_fill(merged, freq=freq, fill=fill)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, index=False, encoding="utf-8-sig")
    return prepared


def add_labels(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    theta_quantile: float,
) -> tuple[pd.DataFrame, float]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0 < theta_quantile < 1:
        raise ValueError("theta_quantile must be between 0 and 1")

    labeled = df.copy()
    labeled[target_col] = pd.to_numeric(labeled[target_col], errors="coerce")
    labeled["future_yield_change"] = labeled[target_col].shift(-horizon) - labeled[target_col]
    labeled = labeled.dropna(subset=[target_col, "future_yield_change"]).copy()

    changes = labeled["future_yield_change"].astype(float)
    theta = float(changes.abs().quantile(theta_quantile))
    if not np.isfinite(theta):
        raise ValueError("Could not compute label threshold.")

    labeled["label"] = np.select(
        [changes > theta, changes < -theta],
        [0, 1],
        default=2,
    ).astype(np.int64)
    return labeled, theta


def resolve_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    requested: list[str] | None = None,
    exclude_target_feature: bool = False,
) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")
        return requested

    excluded = {"date", "future_yield_change", "label"}
    if exclude_target_feature:
        excluded.add(target_col)
    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return feature_cols


def split_masks(
    dates: pd.Series,
    train_end: str | None = None,
    val_end: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(dates)
    if train_end or val_end:
        if not train_end or not val_end:
            raise ValueError("train_end and val_end must be provided together.")
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
        train_mask = dates <= train_end_ts
        val_mask = (dates > train_end_ts) & (dates <= val_end_ts)
        test_mask = dates > val_end_ts
    else:
        n = len(dates)
        train_cut = max(1, int(n * train_ratio))
        val_cut = max(train_cut + 1, int(n * (train_ratio + val_ratio)))
        val_cut = min(val_cut, n - 1)
        idx = np.arange(n)
        train_mask = idx < train_cut
        val_mask = (idx >= train_cut) & (idx < val_cut)
        test_mask = idx >= val_cut
    return np.asarray(train_mask), np.asarray(val_mask), np.asarray(test_mask)


def make_windows(
    X_scaled: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    future_changes: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(X_scaled) <= window:
        raise ValueError(f"Need more than {window} rows after labeling.")

    X_windows = []
    y_windows = []
    date_windows = []
    change_windows = []
    for end_idx in range(window - 1, len(X_scaled)):
        start_idx = end_idx - window + 1
        X_windows.append(X_scaled[start_idx : end_idx + 1])
        y_windows.append(y[end_idx])
        date_windows.append(dates[end_idx])
        change_windows.append(future_changes[end_idx])
    return (
        np.asarray(X_windows, dtype=np.float32),
        np.asarray(y_windows, dtype=np.int64),
        np.asarray(date_windows),
        np.asarray(change_windows, dtype=np.float64),
    )


def build_windowed_dataset(
    data_path: str | Path,
    target_col: str,
    horizon: int,
    window: int,
    theta_quantile: float = 0.6,
    train_end: str | None = None,
    val_end: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_cols: list[str] | None = None,
    exclude_target_feature: bool = False,
    derive_features: bool = False,
) -> WindowedDataset:
    raw = load_wide_dataset(data_path)
    raw = raw.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")
    if derive_features:
        raw = add_derived_features(raw)
    labeled, theta = add_labels(raw, target_col=target_col, horizon=horizon, theta_quantile=theta_quantile)
    features = resolve_feature_columns(labeled, target_col, feature_cols, exclude_target_feature)

    train_row_mask, _, _ = split_masks(
        labeled["date"],
        train_end=train_end,
        val_end=val_end,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    if int(train_row_mask.sum()) < window + 5:
        raise ValueError("Training period is too short for the requested window.")

    X_rows = labeled[features].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    scaler = StandardScaler()
    scaler.fit(X_rows.loc[train_row_mask].to_numpy(dtype=np.float64))
    X_scaled = scaler.transform(X_rows.to_numpy(dtype=np.float64)).astype(np.float32)

    X_all, y_all, dates_all, changes_all = make_windows(
        X_scaled=X_scaled,
        y=labeled["label"].to_numpy(dtype=np.int64),
        dates=labeled["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        future_changes=labeled["future_yield_change"].to_numpy(dtype=np.float64),
        window=window,
    )
    train_mask, val_mask, test_mask = split_masks(
        pd.Series(pd.to_datetime(dates_all)),
        train_end=train_end,
        val_end=val_end,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    if min(int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("Train/validation/test split produced an empty split.")

    return WindowedDataset(
        X_train=X_all[train_mask],
        y_train=y_all[train_mask],
        X_val=X_all[val_mask],
        y_val=y_all[val_mask],
        X_test=X_all[test_mask],
        y_test=y_all[test_mask],
        dates_train=dates_all[train_mask],
        dates_val=dates_all[val_mask],
        dates_test=dates_all[test_mask],
        changes_train=changes_all[train_mask],
        changes_val=changes_all[val_mask],
        changes_test=changes_all[test_mask],
        feature_cols=features,
        scaler=scaler,
        theta=theta,
        target_col=target_col,
        horizon=horizon,
        window=window,
    )


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> StandardScaler:
    return joblib.load(path)
