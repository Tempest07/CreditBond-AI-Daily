from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TENORS = [
    {
        "name": "AAA3Y",
        "target_col": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):3\u5e74",
        "duration": "2.8",
        "out_dir": "curve_2020_AAA3Y_h5",
    },
    {
        "name": "AAA5Y",
        "target_col": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):5\u5e74",
        "duration": "4.5",
        "out_dir": "curve_2020_AAA5Y_h5",
    },
    {
        "name": "AAA10Y",
        "target_col": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA):10\u5e74",
        "duration": "7.5",
        "out_dir": "curve_2020_AAA10Y_h5",
    },
    {
        "name": "AAAp20Y",
        "target_col": "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u5230\u671f\u6536\u76ca\u7387(AAA+):20\u5e74",
        "duration": "12.0",
        "out_dir": "curve_2020_AAAp20Y_h5",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CreditCurveNet for all 2020+ curve tenors.")
    parser.add_argument("--data", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--models-root", default="models/credit_curve_net")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", default="80")
    parser.add_argument("--batch-size", default="256")
    parser.add_argument("--horizon", default="5")
    parser.add_argument("--window", default="60")
    parser.add_argument("--theta-quantile", default="0.6")
    parser.add_argument("--hidden-size", default="192")
    parser.add_argument("--regimes", default="4")
    parser.add_argument("--dropout", default="0.15")
    parser.add_argument("--conv-kernels", default="3,7,15")
    parser.add_argument("--lr", default="0.0008")
    parser.add_argument("--patience", default="12")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--yield-unit", default="percent", choices=["percent", "bp"])
    parser.add_argument("--only", choices=[item["name"] for item in TENORS], action="append")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Feature file not found: {data_path}")

    selected = TENORS
    if args.only:
        wanted = set(args.only)
        selected = [item for item in TENORS if item["name"] in wanted]

    for index, item in enumerate(selected):
        out_dir = Path(args.models_root) / item["out_dir"] / "v1_custom_credit_curve_net"
        cmd = [
            sys.executable,
            "scripts/train_credit_curve_net.py",
            "--data",
            str(data_path),
            "--target-col",
            item["target_col"],
            "--horizon",
            args.horizon,
            "--window",
            args.window,
            "--theta-quantile",
            args.theta_quantile,
            "--epochs",
            args.epochs,
            "--batch-size",
            args.batch_size,
            "--hidden-size",
            args.hidden_size,
            "--regimes",
            args.regimes,
            "--dropout",
            args.dropout,
            "--conv-kernels",
            args.conv_kernels,
            "--lr",
            args.lr,
            "--patience",
            args.patience,
            "--device",
            args.device,
            "--duration",
            item["duration"],
            "--yield-unit",
            args.yield_unit,
            "--seed",
            str(int(args.seed) + index),
            "--exclude-target-feature",
            "--out-dir",
            str(out_dir),
        ]
        if args.no_amp:
            cmd.append("--no-amp")
        print(f"\n=== Training CreditCurveNet {item['name']} -> {out_dir} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
