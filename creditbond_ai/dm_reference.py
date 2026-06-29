from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_EDB_LEVELS_PATH = Path("references/dm_api/edb_indicator_levels.csv")


def load_edb_levels(path: str | Path = DEFAULT_EDB_LEVELS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = ["edb_level_id", "分类体系", "一级分类", "二级分类", "三级分类", "四级分类", "五级分类"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"EDB层级字典缺少列: {missing}")
    return df


def search_edb_levels(
    keyword: str,
    path: str | Path = DEFAULT_EDB_LEVELS_PATH,
    max_rows: int = 50,
) -> pd.DataFrame:
    if not keyword:
        raise ValueError("keyword不能为空")
    df = load_edb_levels(path)
    text_cols = [c for c in df.columns if c != "edb_level_id"]
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask = mask | df[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
    result = df.loc[mask].copy()
    if result.empty:
        result["层级路径"] = []
    else:
        result["层级路径"] = result[text_cols].fillna("").astype(str).agg(" > ".join, axis=1).str.replace(r"( > )+$", "", regex=True)
    cols = ["edb_level_id", "层级路径"] + text_cols
    return result[cols].head(max_rows)
