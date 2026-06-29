from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import creditbond_ai  # noqa: F401
import torch

from creditbond_ai.credit_curve_net import (
    CreditCurveNetConfig,
    CreditCurveTrainConfig,
    infer_tenor_index,
    train_credit_curve_net,
)
from creditbond_ai.data import build_windowed_dataset
from creditbond_ai.feature_selection import load_feature_list


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("conv kernels must not be empty.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom CreditCurveNet model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--theta-quantile", type=float, default=0.6)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--exclude-target-feature", action="store_true")
    parser.add_argument("--derive-features", action="store_true")
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--conv-kernels", default="3,7,15")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--yield-unit", choices=["percent", "bp"], default="percent")
    parser.add_argument("--aux-change-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    feature_cols = load_feature_list(args.features_file) if args.features_file else None
    dataset = build_windowed_dataset(
        data_path=args.data,
        target_col=args.target_col,
        horizon=args.horizon,
        window=args.window,
        theta_quantile=args.theta_quantile,
        train_end=args.train_end,
        val_end=args.val_end,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_cols=feature_cols,
        exclude_target_feature=args.exclude_target_feature,
        derive_features=args.derive_features,
    )

    model_config = CreditCurveNetConfig(
        input_size=len(dataset.feature_cols),
        window=args.window,
        hidden_size=args.hidden_size,
        regime_count=args.regimes,
        dropout=args.dropout,
        conv_kernels=parse_int_list(args.conv_kernels),
        tenor_index=infer_tenor_index(args.target_col),
    )
    train_config = CreditCurveTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        class_weight=not args.no_class_weight,
        duration=args.duration,
        yield_unit=args.yield_unit,
        aux_change_weight=args.aux_change_weight,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
    )

    metrics = train_credit_curve_net(dataset, args.out_dir, model_config, train_config)
    summary = {
        "out_dir": str(Path(args.out_dir)),
        "target_col": args.target_col,
        "horizon": args.horizon,
        "window": args.window,
        "features": len(dataset.feature_cols),
        "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0),
        "test_accuracy": metrics.get("classification_report", {}).get("accuracy", 0.0),
        "test_macro_f1": metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0),
        "proxy_return": metrics.get("backtest", {}).get("total_return_proxy", 0.0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
