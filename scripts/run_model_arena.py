from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.model_arena import run_model_arena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the credit-bond model arena report.")
    parser.add_argument("--features", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--out-dir", default="data/model_arena/curve_2020_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = run_model_arena(
        features_path=args.features,
        models_root=args.models_root,
        out_dir=args.out_dir,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
