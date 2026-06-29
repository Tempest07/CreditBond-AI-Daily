from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.probability_calibration import run_probability_calibration_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probability calibration experiment for model probabilities.")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--out-dir", default="data/model_arena/curve_2020_probability_calibration_v1")
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = run_probability_calibration_experiment(
        models_root=args.models_root,
        out_dir=args.out_dir,
        calibration_ratio=args.calibration_ratio,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
