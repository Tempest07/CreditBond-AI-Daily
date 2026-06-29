from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CoveragePaths:
    config_path: Path
    raw_dir: Path
    out_dir: Path
    model_features_path: Path | None = None


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _parse_dm_data_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        max_value = numeric.dropna().abs().max()
        unit = "ms" if max_value > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _classify_indicator(alias: str, role: str, notes: str) -> str:
    text = f"{alias} {role} {notes}"
    if "M2" in text or "社融" in text or "社会融资" in text or "贷款" in text or "PMI" in text:
        return "宏观"
    if "中短期票据" in text or "信用" in text:
        return "信用债"
    if "国债" in text or "国开" in text or "利率债" in text:
        return "利率债"
    if "资金" in text or "回购" in text or "DR" in text or "R007" in text or "存单" in text:
        return "资金面"
    return "其他"


def _tenor_from_alias(alias: str) -> str:
    for tenor in ("1年", "3年", "5年", "7年", "10年", "15年", "20年", "30年"):
        if tenor in alias:
            return tenor
    return ""


def _coverage_level(years: float | None) -> str:
    if years is None:
        return "无数据"
    if years < 1:
        return "严重短板：不足1年"
    if years < 3:
        return "短板：不足3年"
    if years < 5:
        return "一般：3至5年"
    return "较好：5年以上"


def _fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_dt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _summarize_raw_file(raw_path: Path) -> dict[str, Any]:
    if not raw_path.exists():
        return {
            "raw_exists": False,
            "raw_rows": 0,
            "valid_rows": 0,
            "unique_dates": 0,
            "actual_start": None,
            "actual_end": None,
            "first_publish_time": None,
            "last_publish_time": None,
            "unit": "",
            "data_source": "",
            "dm_indicator_name": "",
        }
    raw = _read_csv(raw_path)
    if raw.empty:
        return {
            "raw_exists": True,
            "raw_rows": 0,
            "valid_rows": 0,
            "unique_dates": 0,
            "actual_start": None,
            "actual_end": None,
            "first_publish_time": None,
            "last_publish_time": None,
            "unit": "",
            "data_source": "",
            "dm_indicator_name": "",
        }
    date_col = "data_date" if "data_date" in raw.columns else None
    value_col = "data_value" if "data_value" in raw.columns else None
    dates = _parse_dm_data_date(raw[date_col]) if date_col else pd.Series(dtype="datetime64[ns]")
    values = pd.to_numeric(raw[value_col], errors="coerce") if value_col else pd.Series(dtype="float64")
    valid = dates.notna() & values.notna()
    publish = pd.to_datetime(raw["publish_time"], errors="coerce") if "publish_time" in raw.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "raw_exists": True,
        "raw_rows": int(len(raw)),
        "valid_rows": int(valid.sum()),
        "unique_dates": int(dates[valid].dt.normalize().nunique()) if valid.any() else 0,
        "actual_start": dates[valid].min() if valid.any() else None,
        "actual_end": dates[valid].max() if valid.any() else None,
        "first_publish_time": publish.min() if not publish.empty else None,
        "last_publish_time": publish.max() if not publish.empty else None,
        "unit": str(raw["basic_indicator_unit"].dropna().iloc[0]) if "basic_indicator_unit" in raw.columns and raw["basic_indicator_unit"].notna().any() else "",
        "data_source": str(raw["data_source"].dropna().iloc[0]) if "data_source" in raw.columns and raw["data_source"].notna().any() else "",
        "dm_indicator_name": str(raw["indicator_name"].dropna().iloc[0]) if "indicator_name" in raw.columns and raw["indicator_name"].notna().any() else "",
    }


def build_dm_coverage(paths: CoveragePaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = _read_csv(paths.config_path)
    required = {"indicator_id", "alias"}
    missing = required.difference(config.columns)
    if missing:
        raise ValueError(f"配置文件缺少字段：{sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, cfg in config.iterrows():
        indicator_id = str(cfg.get("indicator_id", "")).strip()
        alias = str(cfg.get("alias", "")).strip()
        raw_path = paths.raw_dir / f"{indicator_id}.csv"
        raw = _summarize_raw_file(raw_path)
        actual_start = raw["actual_start"]
        actual_end = raw["actual_end"]
        coverage_days = None
        coverage_years = None
        if actual_start is not None and actual_end is not None and not pd.isna(actual_start) and not pd.isna(actual_end):
            coverage_days = int((pd.Timestamp(actual_end).normalize() - pd.Timestamp(actual_start).normalize()).days) + 1
            coverage_years = coverage_days / 365.25
        configured_start = pd.to_datetime(cfg.get("start_date", None), errors="coerce")
        start_gap_days = None
        if not pd.isna(configured_start) and actual_start is not None and not pd.isna(actual_start):
            start_gap_days = int((pd.Timestamp(actual_start).normalize() - configured_start.normalize()).days)
        row = {
            "enabled": cfg.get("enabled", ""),
            "role": cfg.get("role", ""),
            "category": _classify_indicator(alias, str(cfg.get("role", "")), str(cfg.get("notes", ""))),
            "tenor": _tenor_from_alias(alias),
            "indicator_id": indicator_id,
            "alias": alias,
            "frequency": cfg.get("frequency", ""),
            "configured_start_date": _fmt_date(configured_start),
            "actual_start_date": _fmt_date(actual_start),
            "actual_end_date": _fmt_date(actual_end),
            "coverage_days": coverage_days,
            "coverage_years": coverage_years,
            "coverage_level": _coverage_level(coverage_years),
            "start_gap_days_vs_config": start_gap_days,
            "valid_rows": raw["valid_rows"],
            "unique_dates": raw["unique_dates"],
            "raw_rows": raw["raw_rows"],
            "raw_exists": raw["raw_exists"],
            "unit": raw["unit"],
            "data_source": raw["data_source"],
            "first_publish_time": _fmt_dt(raw["first_publish_time"]),
            "last_publish_time": _fmt_dt(raw["last_publish_time"]),
            "dm_indicator_name": raw["dm_indicator_name"],
            "notes": cfg.get("notes", ""),
            "raw_file": str(raw_path),
        }
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail = detail.sort_values(
        by=["coverage_years", "category", "alias"],
        ascending=[True, True, True],
        na_position="first",
    ).reset_index(drop=True)

    model_info: dict[str, Any] = {}
    if paths.model_features_path and paths.model_features_path.exists():
        features = _read_csv(paths.model_features_path)
        if not features.empty and "date" in features.columns:
            dates = pd.to_datetime(features["date"], errors="coerce").dropna()
            model_info = {
                "model_features_path": str(paths.model_features_path),
                "model_rows": int(len(features)),
                "model_columns": int(len(features.columns)),
                "model_start": _fmt_date(dates.min()) if not dates.empty else "",
                "model_end": _fmt_date(dates.max()) if not dates.empty else "",
            }

    overview = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(paths.config_path),
        "raw_dir": str(paths.raw_dir),
        "indicator_count": int(len(detail)),
        "raw_missing_count": int((~detail["raw_exists"].astype(bool)).sum()),
        "shortest": detail.head(10).to_dict(orient="records"),
        "by_category": detail.groupby("category", dropna=False).agg(
            指标数量=("indicator_id", "count"),
            最早开始=("actual_start_date", "min"),
            最晚开始=("actual_start_date", "max"),
            最短年限=("coverage_years", "min"),
            中位年限=("coverage_years", "median"),
        ).reset_index().to_dict(orient="records"),
        "model_info": model_info,
    }
    return detail, overview


def _write_html(detail: pd.DataFrame, overview: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shortest = detail.head(8)
    category = pd.DataFrame(overview["by_category"])

    def table_rows(df: pd.DataFrame, cols: list[str]) -> str:
        rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{html.escape(_fmt_float(row[col]) if isinstance(row.get(col), float) else str(row.get(col, '') if row.get(col, '') is not None else ''))}</td>" for col in cols)
            rows.append(f"<tr>{cells}</tr>")
        return "\n".join(rows)

    detail_cols = [
        "category",
        "tenor",
        "alias",
        "indicator_id",
        "actual_start_date",
        "actual_end_date",
        "coverage_years",
        "coverage_level",
        "valid_rows",
        "data_source",
        "notes",
    ]
    category_cols = ["category", "指标数量", "最早开始", "最晚开始", "最短年限", "中位年限"]
    model_info = overview.get("model_info", {})
    model_note = ""
    if model_info:
        model_note = (
            f"模型特征表：{html.escape(model_info.get('model_start', ''))} 至 "
            f"{html.escape(model_info.get('model_end', ''))}，"
            f"{html.escape(str(model_info.get('model_rows', '')))} 行，"
            f"{html.escape(str(model_info.get('model_columns', '')))} 列。"
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DM 数据覆盖期限体检</title>
  <style>
    :root {{
      --ink: #1c2630;
      --muted: #65717e;
      --line: #dce4ea;
      --paper: #fff;
      --back: #f5f7f9;
      --blue: #2f6f9f;
      --red: #b94a48;
      --amber: #b7812f;
      --green: #238a62;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--back);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    .wrap {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    header {{ background: var(--paper); border-bottom: 1px solid var(--line); padding: 30px 0 24px; }}
    .eyebrow {{ color: var(--blue); font-weight: 800; font-size: 14px; margin-bottom: 8px; }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
    p {{ color: var(--muted); margin: 10px 0 0; }}
    main {{ padding: 22px 0 46px; display: grid; gap: 18px; }}
    section {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 20px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ background: #f8fafb; border-top: 3px solid var(--blue); border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #f8fafb; color: var(--muted); font-weight: 700; }}
    td:nth-child(7), td:nth-child(9) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ border-left: 4px solid var(--amber); background: #fff9ef; color: #4b3a23; padding: 12px 14px; border-radius: 6px; }}
    .scroll {{ overflow-x: auto; }}
    @media (max-width: 860px) {{ .metric-grid {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} table {{ white-space: nowrap; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="eyebrow">DM 数据覆盖期限体检</div>
      <h1>最短板优先：先补齐历史，再谈模型升级</h1>
      <p>生成时间：{html.escape(overview["generated_at"])}。本报告基于本地 DM 原始文件和主配置文件生成。</p>
    </div>
  </header>
  <main class="wrap">
    <section>
      <h2>概览</h2>
      <div class="metric-grid">
        <div class="metric"><span>指标数量</span><strong>{len(detail)}</strong></div>
        <div class="metric"><span>最短实际起点</span><strong>{html.escape(str(detail["actual_start_date"].min()))}</strong></div>
        <div class="metric"><span>最长实际终点</span><strong>{html.escape(str(detail["actual_end_date"].max()))}</strong></div>
        <div class="metric"><span>原始文件缺失</span><strong>{html.escape(str(overview["raw_missing_count"]))}</strong></div>
      </div>
      <p>{model_note}</p>
    </section>
    <section>
      <h2>最短覆盖指标</h2>
      <div class="note">这些是优先要去 DM 里找更长历史、替代口径或补充来源的指标。</div>
      <div class="scroll" style="margin-top: 12px;">
        <table>
          <thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in detail_cols)}</tr></thead>
          <tbody>{table_rows(shortest, detail_cols)}</tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>按类别汇总</h2>
      <div class="scroll">
        <table>
          <thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in category_cols)}</tr></thead>
          <tbody>{table_rows(category, category_cols)}</tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>全部指标明细</h2>
      <div class="scroll">
        <table>
          <thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in detail_cols)}</tr></thead>
          <tbody>{table_rows(detail, detail_cols)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def run_dm_coverage_report(
    config_path: str | Path,
    raw_dir: str | Path,
    out_dir: str | Path,
    model_features_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = CoveragePaths(
        config_path=Path(config_path),
        raw_dir=Path(raw_dir),
        out_dir=Path(out_dir),
        model_features_path=Path(model_features_path) if model_features_path else None,
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    detail, overview = build_dm_coverage(paths)
    detail_csv = paths.out_dir / "dm_coverage_detail.csv"
    overview_json = paths.out_dir / "dm_coverage_summary.json"
    html_path = paths.out_dir / "dm_coverage_report.html"
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    overview_json.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(detail, overview, html_path)
    return {
        "detail_csv": str(detail_csv),
        "summary_json": str(overview_json),
        "html_report": str(html_path),
        "indicator_count": int(len(detail)),
        "shortest": detail.head(10).to_dict(orient="records"),
        "model_info": overview.get("model_info", {}),
    }
