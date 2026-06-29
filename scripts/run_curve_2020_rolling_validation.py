from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creditbond_ai.no_data_experiments import RollingValidationConfig, run_rolling_validation


TARGET_COLS = [
    "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):3\u5e74",
    "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):5\u5e74",
    "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):10\u5e74",
    "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA+):20\u5e74",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling train/predict validation for 2020+ curve data.")
    parser.add_argument("--data", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--out-dir", default="models/curve_2020_rolling_validation")
    parser.add_argument("--models", default="gru,tcn,transformer")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--min-train-rows", type=int, default=504)
    parser.add_argument("--val-rows", type=int, default=126)
    parser.add_argument("--test-rows", type=int, default=126)
    parser.add_argument("--step-rows", type=int, default=126)
    parser.add_argument("--max-folds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--theta-quantile", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--derive-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_rolling_validation(
        RollingValidationConfig(
            data_path=args.data,
            target_cols=TARGET_COLS,
            out_dir=args.out_dir,
            models=[item.strip() for item in args.models.split(",") if item.strip()],
            theta_quantile=args.theta_quantile,
            horizon=args.horizon,
            window=args.window,
            min_train_rows=args.min_train_rows,
            val_rows=args.val_rows,
            test_rows=args.test_rows,
            step_rows=args.step_rows,
            max_folds=args.max_folds,
            exclude_target_feature=True,
            derive_features=args.derive_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            hidden_size=args.hidden_size,
            layers=args.layers,
            dropout=args.dropout,
            heads=args.heads,
            kernel_size=args.kernel_size,
            device=args.device,
            duration_map={"3": 2.8, "5": 4.5, "10": 7.5, "20": 12.0},
            yield_unit="percent",
            seed=args.seed,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
