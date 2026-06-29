from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feature_list(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "feature" in df.columns:
            values = df["feature"].dropna().astype(str).tolist()
        elif "特征" in df.columns:
            values = df["特征"].dropna().astype(str).tolist()
        elif len(df.columns) == 1:
            values = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            raise ValueError("特征CSV需要包含 feature 列、特征列，或只有一列。")
    else:
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def select_features_from_importance(
    importance_path: str | Path,
    out_path: str | Path,
    top_n: int = 40,
    min_drop: float = 0.0,
    include_contains: list[str] | None = None,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n 必须大于等于 1。")
    importance = pd.read_csv(importance_path, encoding="utf-8-sig")
    required = {"feature", "importance_drop"}
    missing = required - set(importance.columns)
    if missing:
        raise ValueError(f"重要性文件缺少列: {sorted(missing)}")

    ranked = importance.sort_values("importance_drop", ascending=False).copy()
    selected = ranked[ranked["importance_drop"] >= min_drop].head(top_n).copy()

    include_contains = include_contains or []
    forced_rows = []
    for token in include_contains:
        token = token.strip()
        if not token:
            continue
        matched = ranked[ranked["feature"].astype(str).str.contains(token, regex=False, na=False)]
        forced_rows.append(matched)
    if forced_rows:
        selected = pd.concat([selected] + forced_rows, ignore_index=True)
        selected = selected.drop_duplicates(subset=["feature"], keep="first")
        selected = selected.sort_values("importance_drop", ascending=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        selected.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        out_path.write_text("\n".join(selected["feature"].astype(str).tolist()) + "\n", encoding="utf-8")
    return selected
