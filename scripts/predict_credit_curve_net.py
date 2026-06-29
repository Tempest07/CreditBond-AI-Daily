from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.credit_curve_net import predict_latest_credit_curve_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict latest signal with a trained CreditCurveNet.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_latest_credit_curve_net(
        model_dir=args.model_dir,
        data_path=args.data,
        out_path=args.out,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
