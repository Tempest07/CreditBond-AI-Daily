from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.dm_api import create_dm_client
from creditbond_ai.dm_intraday import (
    TBOND_ACTIVE_CODES,
    TREASURY_FUTURES_CODES,
    extract_bond_codes,
    fetch_bond_bars,
    fetch_bond_insti_sentiment,
    fetch_bond_realtime_quote,
    fetch_futures_bars,
    fetch_futures_basis,
    fetch_rolling_bonds,
    save_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a small DM intraday market snapshot.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--kline-type", type=int, default=2, help="1=1分钟, 2=5分钟, 3=15分钟.")
    parser.add_argument("--tbond-active-codes", default=",".join(TBOND_ACTIVE_CODES))
    parser.add_argument("--treasury-futures-codes", default=",".join(TREASURY_FUTURES_CODES))
    parser.add_argument("--key-tenors", default="2,5,10,30")
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _split_int_csv(value: str) -> list[int]:
    return [int(float(item.strip())) for item in str(value).split(",") if item.strip()]


def _safe_fetch(name: str, fetcher: Callable[[], pd.DataFrame], out_dir: Path) -> dict:
    row = {"name": name, "ok": False, "rows": 0, "columns": [], "error": ""}
    try:
        df = fetcher()
        save_frame(df, out_dir / f"{name}.csv")
        row["ok"] = True
        row["rows"] = int(len(df))
        row["columns"] = list(df.columns)
    except Exception as exc:
        row["error"] = str(exc)[:1000]
    return row


def main() -> int:
    args = parse_args()
    snapshot_date = str(args.date)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "intraday" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    client = create_dm_client(timeout=30)
    active_codes = _split_csv(args.tbond_active_codes)
    futures_codes = _split_csv(args.treasury_futures_codes)
    key_tenors = _split_int_csv(args.key_tenors)
    summary: list[dict] = []

    summary.append(
        _safe_fetch(
            "tbond_active_bars",
            lambda: fetch_bond_bars(
                active_codes,
                start_datetime=snapshot_date,
                end_datetime=snapshot_date,
                kline_type=args.kline_type,
                client=client,
            ),
            out_dir,
        )
    )

    rolling_df = pd.DataFrame()
    def _fetch_rolling() -> pd.DataFrame:
        nonlocal rolling_df
        rolling_df = fetch_rolling_bonds(
            key_tenors=key_tenors,
            start_date=snapshot_date,
            end_date=snapshot_date,
            sequence_type=1,
            bond_filter_type=1,
            client=client,
        )
        return rolling_df

    summary.append(_safe_fetch("tbond_rolling_active", _fetch_rolling, out_dir))
    actual_codes = extract_bond_codes(rolling_df)
    (out_dir / "tbond_actual_codes.json").write_text(
        json.dumps(actual_codes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if actual_codes:
        summary.append(
            _safe_fetch(
                "tbond_realtime_quote",
                lambda: fetch_bond_realtime_quote(actual_codes, client=client),
                out_dir,
            )
        )
        summary.append(
            _safe_fetch(
                "tbond_actual_bars",
                lambda: fetch_bond_bars(
                    actual_codes[:5],
                    start_datetime=snapshot_date,
                    end_datetime=snapshot_date,
                    kline_type=args.kline_type,
                    client=client,
                ),
                out_dir,
            )
        )

    summary.append(
        _safe_fetch(
            "treasury_futures_bars",
            lambda: fetch_futures_bars(
                futures_codes,
                start_datetime=snapshot_date,
                end_datetime=snapshot_date,
                kline_type=args.kline_type,
                client=client,
            ),
            out_dir,
        )
    )
    summary.append(
        _safe_fetch(
            "treasury_futures_basis",
            lambda: fetch_futures_basis(futures_codes, start_date=snapshot_date, end_date=snapshot_date, client=client),
            out_dir,
        )
    )
    for source in (10, 14):
        summary.append(
            _safe_fetch(
                f"bond_insti_sentiment_{source}",
                lambda source=source: fetch_bond_insti_sentiment(
                    data_source=source,
                    start_datetime=snapshot_date,
                    end_datetime=snapshot_date,
                    freqs=(1,),
                    client=client,
                ),
                out_dir,
            )
        )

    summary_df = pd.DataFrame(summary)
    save_frame(summary_df, out_dir / "snapshot_summary.csv")
    print(summary_df[["name", "ok", "rows", "error"]].to_string(index=False))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
