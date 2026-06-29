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
    parser = argparse.ArgumentParser(description="Train 2020+ curve models for all credit tenors.")
    parser.add_argument("--data", default="data/dm_daily_master_curve_2020/processed/dm_features_latest.csv")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", default="40")
    parser.add_argument("--batch-size", default="128")
    parser.add_argument("--horizon", default="5")
    parser.add_argument("--window", default="60")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--models", default="gru,tcn,transformer")
    parser.add_argument("--yield-unit", default="percent", choices=["percent", "bp"])
    parser.add_argument("--only", choices=[item["name"] for item in TENORS], action="append")
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

    for item in selected:
        out_dir = Path(args.models_root) / item["out_dir"] / "01_full_features"
        cmd = [
            sys.executable,
            "-m",
            "creditbond_ai.cli",
            "compare",
            "--data",
            str(data_path),
            "--target-col",
            item["target_col"],
            "--horizon",
            args.horizon,
            "--window",
            args.window,
            "--epochs",
            args.epochs,
            "--batch-size",
            args.batch_size,
            "--models",
            args.models,
            "--device",
            args.device,
            "--duration",
            item["duration"],
            "--yield-unit",
            args.yield_unit,
            "--seed",
            args.seed,
            "--exclude-target-feature",
            "--out-dir",
            str(out_dir),
        ]
        print(f"\n=== Training {item['name']} -> {out_dir} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
