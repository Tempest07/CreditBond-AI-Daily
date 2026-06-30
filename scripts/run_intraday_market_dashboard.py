from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.intraday_features import build_intraday_features
from creditbond_ai.market_context import render_realtime_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a realtime-style credit-bond intraday market dashboard.")
    parser.add_argument("--snapshot-dir", required=True, help="Directory from run_dm_intraday_snapshot.py.")
    parser.add_argument("--context-dir", required=True, help="Directory from run_dm_market_context.py.")
    parser.add_argument("--out", default="")
    parser.add_argument("--refresh-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    context_dir = Path(args.context_dir)
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"找不到日内快照目录：{snapshot_dir}")
    if not context_dir.exists():
        raise FileNotFoundError(f"找不到市场背景目录：{context_dir}")
    build_intraday_features(snapshot_dir)
    out = Path(args.out) if args.out else context_dir / "reports" / "intraday_market_dashboard.html"
    report = render_realtime_dashboard(
        context_dir=context_dir,
        snapshot_dir=snapshot_dir,
        out_path=out,
        refresh_seconds=args.refresh_seconds,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
