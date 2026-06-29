from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from .dm_api import fetch_edb_indicator_data, parse_dm_datetime


GROUP_LABELS = {
    "cp_AAA": "中短期票据 AAA",
    "cp_AAAplus": "中短期票据 AAA+",
    "cp_AAAminus": "中短期票据 AAA-",
    "ent_AAAminus": "企业债 AAA-",
    "ent_AAplus": "有担保企业债 AA+",
    "ent_A": "企业债 A",
    "cdb_AAA": "政策性金融债(国开) AAA",
    "treasury": "国债",
    "zhongzhai_monthly": "中债国债收益率(月频)",
}

SOURCE_KIND = {
    "zhongzhai_monthly": "非CFETS：中债月频表",
}

COLUMN_ALIASES = {
    "indicatorId": "indicator_id",
    "indicatorName": "indicator_name",
    "dataDate": "data_date",
    "dataValue": "data_value",
    "basicIndicatorUnit": "basic_indicator_unit",
    "statisticalFrequency": "statistical_frequency",
}


def _normalise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in raw.columns})
    if "data_date" in raw.columns:
        raw["data_date"] = parse_dm_datetime(raw["data_date"])
    if "data_value" in raw.columns:
        raw["data_value"] = pd.to_numeric(raw["data_value"], errors="coerce")
    return raw


def _tenor_from_name(name: str) -> float | None:
    text = str(name)
    for term in ["30", "20", "10", "7", "5", "3", "1"]:
        if f":{term}年" in text or text.endswith(f"{term}年"):
            return float(term)
    return None


def _standardise_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in candidates.columns})
    rename_map = {
        "indicatorId": "indicator_id",
        "indicatorName": "indicator_name",
        "freq": "frequency",
        "statisticalFrequency": "frequency",
    }
    candidates = candidates.rename(columns={k: v for k, v in rename_map.items() if k in candidates.columns})
    if "indicator_id" not in candidates.columns:
        raise ValueError("候选曲线文件必须包含 indicator_id 或 indicatorId。")
    if "indicator_name" not in candidates.columns:
        candidates["indicator_name"] = candidates["indicator_id"]
    if "frequency" not in candidates.columns:
        candidates["frequency"] = ""
    if "group" not in candidates.columns:
        candidates["group"] = "unknown"
    if "tenor" not in candidates.columns:
        candidates["tenor"] = candidates["indicator_name"].map(_tenor_from_name)
    candidates["indicator_id"] = candidates["indicator_id"].astype(str).str.strip()
    candidates["indicator_name"] = candidates["indicator_name"].astype(str).str.strip()
    candidates["frequency"] = candidates["frequency"].astype(str).str.strip()
    candidates["group"] = candidates["group"].astype(str).str.strip()
    candidates["tenor"] = pd.to_numeric(candidates["tenor"], errors="coerce")
    candidates = candidates[candidates["indicator_id"].ne("")].copy()
    return candidates


def _load_candidates(
    candidates_path: str | Path,
    zhongzhai_codes_path: str | Path | None = None,
) -> pd.DataFrame:
    candidates = _standardise_candidate_columns(pd.read_csv(candidates_path, encoding="utf-8-sig"))
    frames = [candidates]

    if zhongzhai_codes_path:
        zhongzhai_path = Path(zhongzhai_codes_path)
        if zhongzhai_path.exists():
            zz = _standardise_candidate_columns(pd.read_csv(zhongzhai_path, encoding="utf-8-sig"))
            zz["group"] = "zhongzhai_monthly"
            zz["tenor"] = zz["indicator_name"].map(_tenor_from_name)
            zz = zz[zz["tenor"].isin([3.0, 5.0, 10.0, 20.0, 30.0])].copy()
            frames.append(zz)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["indicator_id"], keep="first").reset_index(drop=True)
    combined["group_label"] = combined["group"].map(GROUP_LABELS).fillna(combined["group"])
    combined["source_kind"] = combined["group"].map(SOURCE_KIND).fillna("DM债券收益率曲线层级：多为CFETS口径")
    return combined


def _summarise_raw(raw_path: Path, candidate: pd.Series) -> dict[str, Any]:
    if not raw_path.exists():
        return {
            "status": "未抓取",
            "rows": 0,
            "actual_start": "",
            "actual_end": "",
            "coverage_years": None,
            "latest_value": None,
            "raw_file": "",
        }
    try:
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    except EmptyDataError:
        return {
            "status": "无返回数据",
            "rows": 0,
            "actual_start": "",
            "actual_end": "",
            "coverage_years": None,
            "latest_value": None,
            "raw_file": str(raw_path),
        }
    raw = _normalise_raw(raw)
    if "data_date" not in raw.columns or raw["data_date"].dropna().empty:
        return {
            "status": "无有效日期",
            "rows": int(len(raw)),
            "actual_start": "",
            "actual_end": "",
            "coverage_years": None,
            "latest_value": None,
            "raw_file": str(raw_path),
        }
    dated = raw.dropna(subset=["data_date"]).sort_values("data_date")
    start = dated["data_date"].min().normalize()
    end = dated["data_date"].max().normalize()
    coverage_years = ((end - start).days + 1) / 365.25
    latest_value = None
    if "data_value" in dated.columns and dated["data_value"].notna().any():
        latest_value = float(dated["data_value"].dropna().iloc[-1])
    return {
        "status": "已抓取",
        "rows": int(len(dated)),
        "actual_start": start.strftime("%Y-%m-%d"),
        "actual_end": end.strftime("%Y-%m-%d"),
        "coverage_years": round(float(coverage_years), 2),
        "latest_value": latest_value,
        "raw_file": str(raw_path),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "" or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_html(detail: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    fetched = detail[detail["status"].eq("已抓取")].copy()
    if not fetched.empty:
        fetched["coverage_sort"] = pd.to_numeric(fetched["coverage_years"], errors="coerce")
        longest = fetched.sort_values(["coverage_sort", "rows"], ascending=False).iloc[0]
        shortest = fetched.sort_values(["coverage_sort", "rows"], ascending=True).iloc[0]
    else:
        longest = shortest = None

    group_rows = []
    if not fetched.empty:
        grouped = fetched.groupby(["group_label", "source_kind"], dropna=False)
        for (group_label, source_kind), part in grouped:
            years = pd.to_numeric(part["coverage_years"], errors="coerce")
            group_rows.append(
                {
                    "group_label": group_label,
                    "source_kind": source_kind,
                    "count": int(len(part)),
                    "min_start": part["actual_start"].min(),
                    "max_end": part["actual_end"].max(),
                    "min_years": years.min(),
                    "max_years": years.max(),
                }
            )
    group_df = pd.DataFrame(group_rows)

    def esc(value: Any, digits: int = 2) -> str:
        return html.escape(_fmt(value, digits=digits))

    detail_rows = []
    for _, row in detail.sort_values(["status", "source_kind", "group_label", "tenor", "indicator_id"]).iterrows():
        detail_rows.append(
            "<tr>"
            f"<td>{esc(row.get('source_kind'))}</td>"
            f"<td>{esc(row.get('group_label'))}</td>"
            f"<td>{esc(row.get('tenor'))}</td>"
            f"<td>{esc(row.get('indicator_id'))}</td>"
            f"<td>{esc(row.get('indicator_name'))}</td>"
            f"<td>{esc(row.get('frequency'))}</td>"
            f"<td>{esc(row.get('rows'))}</td>"
            f"<td>{esc(row.get('actual_start'))}</td>"
            f"<td>{esc(row.get('actual_end'))}</td>"
            f"<td>{esc(row.get('coverage_years'))}</td>"
            f"<td>{esc(row.get('latest_value'), 4)}</td>"
            f"<td>{esc(row.get('status'))}</td>"
            "</tr>"
        )

    group_html = ""
    if not group_df.empty:
        for _, row in group_df.sort_values(["source_kind", "group_label"]).iterrows():
            group_html += (
                "<tr>"
                f"<td>{esc(row['source_kind'])}</td>"
                f"<td>{esc(row['group_label'])}</td>"
                f"<td>{esc(row['count'])}</td>"
                f"<td>{esc(row['min_start'])}</td>"
                f"<td>{esc(row['max_end'])}</td>"
                f"<td>{esc(row['min_years'])}</td>"
                f"<td>{esc(row['max_years'])}</td>"
                "</tr>"
            )

    cards = [
        ("候选曲线", f"{summary['candidate_count']} 条"),
        ("已抓取", f"{summary['fetched_count']} 条"),
        ("请求区间", f"{summary['requested_start']} 至 {summary['requested_end']}"),
    ]
    if longest is not None:
        cards.append(("最长历史", f"{longest['actual_start']} 起，{_fmt(longest['coverage_years'])} 年"))
    if shortest is not None:
        cards.append(("最短历史", f"{shortest['actual_start']} 起，{_fmt(shortest['coverage_years'])} 年"))

    card_html = "".join(
        f"<section class='card'><div>{html.escape(title)}</div><strong>{html.escape(value)}</strong></section>"
        for title, value in cards
    )

    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DM 曲线历史长度审计</title>
<style>
:root {{
  --ink: #17202a;
  --muted: #687385;
  --line: #d8dde6;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --accent: #2764c4;
  --accent-soft: #e8f0ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  padding: 28px 36px 18px;
  background: #ffffff;
  border-bottom: 1px solid var(--line);
}}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
p {{ margin: 0; color: var(--muted); line-height: 1.7; }}
main {{ padding: 24px 36px 42px; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}}
.card div {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
.card strong {{ font-size: 20px; }}
.note {{
  background: var(--accent-soft);
  border-left: 4px solid var(--accent);
  padding: 14px 16px;
  margin: 18px 0;
  line-height: 1.7;
}}
section.block {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  margin-top: 18px;
}}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
.table-wrap {{ overflow-x: auto; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
  text-align: left;
  white-space: nowrap;
}}
th {{
  background: #f0f3f8;
  color: #354052;
  position: sticky;
  top: 0;
}}
td:nth-child(5) {{ min-width: 270px; white-space: normal; }}
.footer {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>DM 曲线历史长度审计</h1>
  <p>用于确认不同信用债/利率债口径在 DM API 中实际能取到多长历史。日期为 API 返回数据日期，不是本地文件创建日期。</p>
</header>
<main>
  <div class="cards">{card_html}</div>
  <div class="note">
    本报告把“中短票、企业债、国开、国债”等同期限曲线与“中债国债收益率(月频)”分开标注。
    如果某条曲线从 2024-09-18 才开始，含义是 DM API 在本次请求区间内实际只返回这些日期。
  </div>
  <section class="block">
    <h2>按口径汇总</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>来源口径</th><th>曲线组</th><th>条数</th><th>最早日期</th><th>最晚日期</th><th>最短年限</th><th>最长年限</th></tr></thead>
        <tbody>{group_html}</tbody>
      </table>
    </div>
  </section>
  <section class="block">
    <h2>逐条明细</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>来源口径</th><th>曲线组</th><th>期限</th><th>指标ID</th><th>指标名称</th><th>频率</th><th>行数</th><th>开始</th><th>结束</th><th>年限</th><th>最新值</th><th>状态</th></tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </div>
  </section>
  <div class="footer">生成文件：{html.escape(str(output_path))}</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def run_curve_history_audit(
    candidates_path: str | Path,
    out_dir: str | Path,
    start_date: str = "2011-01-01",
    end_date: str = "2026-06-24",
    zhongzhai_codes_path: str | Path | None = None,
    offset: int = 0,
    limit: int | None = None,
    refresh: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(candidates_path, zhongzhai_codes_path=zhongzhai_codes_path)
    fetch_slice = candidates.iloc[offset : None if limit is None else offset + limit]
    errors: dict[str, str] = {}

    for _, row in fetch_slice.iterrows():
        indicator_id = str(row["indicator_id"])
        raw_path = raw_dir / f"{indicator_id}.csv"
        if raw_path.exists() and not refresh:
            continue
        try:
            data = fetch_edb_indicator_data(
                indicator_id=indicator_id,
                start_date=start_date,
                end_date=end_date,
                frequency=str(row.get("frequency") or ""),
                timeout=timeout,
            )
            data.to_csv(raw_path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # pragma: no cover - network/API defensive path
            errors[indicator_id] = str(exc)

    detail_rows = []
    for _, row in candidates.iterrows():
        raw_path = raw_dir / f"{row['indicator_id']}.csv"
        summary = _summarise_raw(raw_path, row)
        if row["indicator_id"] in errors:
            summary["status"] = "抓取失败"
            summary["error"] = errors[row["indicator_id"]]
        detail_rows.append({**row.to_dict(), **summary})

    detail = pd.DataFrame(detail_rows)
    detail_path = out_dir / "alternative_curve_history_detail.csv"
    summary_path = out_dir / "alternative_curve_history_summary.json"
    html_path = out_dir / "alternative_curve_history_report.html"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    report_summary = {
        "candidate_count": int(len(candidates)),
        "fetched_count": int(detail["status"].eq("已抓取").sum()),
        "failed_count": int(detail["status"].eq("抓取失败").sum()),
        "requested_start": start_date,
        "requested_end": end_date,
        "offset": int(offset),
        "limit": limit,
        "detail_csv": str(detail_path),
        "html_report": str(html_path),
        "raw_dir": str(raw_dir),
        "errors": errors,
    }
    summary_path.write_text(json.dumps(report_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(detail, report_summary, html_path)
    return report_summary
