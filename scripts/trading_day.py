from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether to run the morning credit-bond report.")
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--holiday-file", default="configs/china_market_holidays.csv")
    return parser.parse_args()


def load_holidays(path: str | Path) -> set[date]:
    holiday_path = Path(path)
    if not holiday_path.is_absolute():
        holiday_path = ROOT / holiday_path
    if not holiday_path.exists():
        return set()
    holidays: set[date] = set()
    with holiday_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            value = str(row.get("date", "")).strip()
            if value:
                holidays.add(date.fromisoformat(value))
    return holidays


def is_trading_day(day: date, holidays: set[date]) -> bool:
    return day.weekday() < 5 and day not in holidays


def previous_trading_day(day: date, holidays: set[date]) -> date:
    cursor = day - timedelta(days=1)
    for _ in range(20):
        if is_trading_day(cursor, holidays):
            return cursor
        cursor -= timedelta(days=1)
    raise RuntimeError(f"Cannot find previous trading day before {day.isoformat()}.")


def main() -> int:
    args = parse_args()
    today = date.fromisoformat(args.today)
    holidays = load_holidays(args.holiday_file)
    should_run = is_trading_day(today, holidays)
    result = {
        "today": today.isoformat(),
        "is_trading_day": should_run,
        "data_end": previous_trading_day(today, holidays).isoformat() if should_run else "",
        "holiday_file": args.holiday_file,
        "skip_reason": "" if should_run else "not_a_trading_day",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
