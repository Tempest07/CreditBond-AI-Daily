from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.market_context import fetch_market_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch DM market context for credit-bond intraday and daily reports.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--primary-lookback-days", type=int, default=30)
    parser.add_argument("--credit-quote-limit", type=int, default=80)
    parser.add_argument("--credit-watchlist", default="")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "market_context" / str(args.date)
    result = fetch_market_context(
        as_of_date=args.date,
        out_dir=out_dir,
        history_days=args.history_days,
        primary_lookback_days=args.primary_lookback_days,
        credit_quote_limit=args.credit_quote_limit,
        credit_watchlist=args.credit_watchlist or None,
        timeout=args.timeout,
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
