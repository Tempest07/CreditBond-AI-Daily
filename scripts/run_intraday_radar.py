from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.intraday_features import build_intraday_features, render_intraday_radar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intraday credit-bond market features and a radar report.")
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory created by run_dm_intraday_snapshot.py, such as data/intraday/2026-06-30.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

    outputs = build_intraday_features(snapshot_dir)
    report_path = render_intraday_radar(snapshot_dir)

    print("日内特征已生成：")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"日内雷达报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
