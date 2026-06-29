from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_windowed_dataset, merge_exported_files, prepare_wide_data
from .cleaning import clean_data_folder, clean_dataset
from .daily import run_daily_dm_update
from .dictionary import create_data_dictionary, validate_data_dictionary
from .dm_coverage import run_dm_coverage_report
from .dm_curve_audit import run_curve_history_audit
from .dm_api import (
    check_dm_environment,
    credentials_help,
    edb_raw_to_point_in_time_wide,
    fetch_bond_yield_curve_data,
    fetch_edb_data_from_config,
    fetch_edb_indicator_codes,
    fetch_edb_indicator_data,
    save_dataframe,
)
from .dm_reference import search_edb_levels
from .feature_selection import load_feature_list, select_features_from_importance
from .importance import model_feature_importance
from .demo import make_demo_dataset
from .model import ModelConfig
from .no_data_experiments import (
    LabelSweepConfig,
    RollingValidationConfig,
    StrategySweepConfig,
    decode_cli_text_list,
    parse_float_list,
    parse_str_list,
    run_label_threshold_sweep,
    run_rolling_validation,
    run_strategy_sweep,
)
from .position_backtest import PositionBacktestConfig, parse_duration_map, run_position_backtest
from .predict import predict_latest
from .training import TrainConfig, train_model


MODEL_CHOICES = ["lstm", "lstm_attention", "gru", "tcn", "transformer"]


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-col", required=True)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--yield-unit", choices=["percent", "bp"], default="percent")
    parser.add_argument("--seed", type=int, default=42)


def build_dataset_from_args(args: argparse.Namespace):
    feature_cols = load_feature_list(args.features_file) if args.features_file else None
    return build_windowed_dataset(
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


def model_config_from_args(args: argparse.Namespace, input_size: int, architecture: str) -> ModelConfig:
    return ModelConfig(
        input_size=input_size,
        architecture=architecture,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        n_heads=args.heads,
        tcn_kernel_size=args.kernel_size,
        max_window=max(args.window, 512),
    )


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        class_weight=not args.no_class_weight,
        duration=args.duration,
        yield_unit=args.yield_unit,
    )


def summarize_metrics(architecture: str, metrics: dict, out_dir: Path) -> dict:
    report = metrics["classification_report"]
    backtest = metrics["backtest"]
    return {
        "model": architecture,
        "out_dir": str(out_dir),
        "accuracy": report.get("accuracy", 0.0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "bullish_f1": report.get("看多", {}).get("f1-score", 0.0),
        "bearish_f1": report.get("看空", {}).get("f1-score", 0.0),
        "range_f1": report.get("震荡", {}).get("f1-score", 0.0),
        "total_return_proxy": backtest.get("total_return_proxy", 0.0),
        "positive_active_ratio": backtest.get("positive_active_ratio", 0.0),
        "active_signal_ratio": backtest.get("active_signal_ratio", 0.0),
    }


def parse_model_list(value: str) -> list[str]:
    requested_models = [m.strip().lower() for m in value.split(",") if m.strip()]
    invalid = [m for m in requested_models if m not in MODEL_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")
    return requested_models


def run_model_comparison(
    args: argparse.Namespace,
    out_dir: str | Path,
    requested_models: list[str],
    features_file: str | Path | None = None,
) -> pd.DataFrame:
    run_args = copy.copy(args)
    run_args.features_file = str(features_file) if features_file else None

    dataset = build_dataset_from_args(run_args)
    rows = []
    base_out = Path(out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    for index, architecture in enumerate(requested_models):
        seed = args.seed + index
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_out = base_out / architecture
        print(f"\n===== training {architecture} =====")
        model_config = model_config_from_args(run_args, len(dataset.feature_cols), architecture)
        train_config = train_config_from_args(run_args)
        metrics = train_model(dataset, model_out, model_config, train_config)
        rows.append(summarize_metrics(architecture, metrics, model_out))

    comparison = pd.DataFrame(rows).sort_values(["macro_f1", "total_return_proxy"], ascending=False)
    comparison_path = base_out / "comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False))
    print(f"comparison saved to {comparison_path}")
    return comparison


def best_model_dir(comparison: pd.DataFrame) -> Path:
    if comparison.empty:
        raise ValueError("comparison is empty.")
    return Path(str(comparison.iloc[0]["out_dir"]))


def write_auto_research_report(
    out_path: str | Path,
    target_col: str,
    horizon: int,
    window: int,
    full_comparison: pd.DataFrame,
    selected_comparison: pd.DataFrame,
    positive_comparison: pd.DataFrame | None,
    first_features_path: Path,
    second_features_path: Path | None,
    first_importance_path: Path,
    second_importance_path: Path | None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, comparison, feature_count in [
        ("全特征", full_comparison, None),
        ("第一轮筛选", selected_comparison, len(load_feature_list(first_features_path))),
        (
            "第二轮正贡献筛选",
            positive_comparison,
            len(load_feature_list(second_features_path)) if second_features_path and second_features_path.exists() else None,
        ),
    ]:
        if comparison is None or comparison.empty:
            continue
        best = comparison.iloc[0]
        rows.append(
            "| {name} | {feature_count} | {model} | {accuracy:.4f} | {macro_f1:.4f} | {ret:.4f} | {active:.4f} |".format(
                name=name,
                feature_count="" if feature_count is None else feature_count,
                model=best["model"],
                accuracy=float(best["accuracy"]),
                macro_f1=float(best["macro_f1"]),
                ret=float(best["total_return_proxy"]),
                active=float(best["active_signal_ratio"]),
            )
        )

    table = "\n".join(rows)
    content = f"""# 自动选指标研究报告

## 基本设置

- 目标列：`{target_col}`
- 预测跨度：未来 {horizon} 个交易日
- 训练窗口：{window} 个交易日

## 最佳结果对比

| 版本 | 特征数 | 最佳模型 | 准确率 | 宏平均 F1 | 收益代理 | 活跃信号占比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{table}

## 文件位置

- 第一轮重要性：`{first_importance_path}`
- 第一轮特征清单：`{first_features_path}`
"""
    if second_importance_path is not None:
        content += f"- 第二轮重要性：`{second_importance_path}`\n"
    if second_features_path is not None:
        content += f"- 第二轮特征清单：`{second_features_path}`\n"

    content += """
## 解读方法

准确率和宏平均 F1 用来看方向判断是否稳定；收益代理和活跃信号占比用来看它是否适合做决策辅助。分类指标更高不一定代表决策效果更好，后续正式版要同时考察收益、回撤、换手和信号覆盖率。
"""
    out_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creditbond-ai",
        description="Local PyTorch toolkit for credit-bond trend prediction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("make-demo", help="Create a synthetic dataset for smoke tests.")
    demo.add_argument("--out", default="data/demo_creditbond.csv")
    demo.add_argument("--days", type=int, default=1800)
    demo.add_argument("--seed", type=int, default=7)

    prep = sub.add_parser("prepare-wide", help="Normalize one wide CSV with date + factor columns.")
    prep.add_argument("--input", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--freq", default="B")
    prep.add_argument("--fill", choices=["ffill", "interpolate", "none"], default="ffill")
    prep.add_argument("--derive-features", action="store_true")

    merge = sub.add_parser("merge-exports", help="Merge one-factor-per-file legacy CSV exports.")
    merge.add_argument("--input-dir", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--mapping", default=None, help="Optional JSON filename/stem to output-column mapping.")
    merge.add_argument("--freq", default="B")
    merge.add_argument("--fill", choices=["ffill", "interpolate", "none"], default="ffill")

    clean = sub.add_parser("clean-data", help="Run the standard data cleaning pipeline before modeling.")
    clean.add_argument("--input", required=True)
    clean.add_argument("--out-dir", default="data/cleaned")
    clean.add_argument("--name", default=None)
    clean.add_argument("--freq", default="B")
    clean.add_argument("--fill", choices=["ffill", "interpolate", "none"], default="ffill")
    clean.add_argument("--no-derived-features", action="store_true")
    clean.add_argument("--raw-metadata", default=None)

    clean_batch = sub.add_parser("clean-batch", help="Clean many similarly structured CSV files and merge them.")
    clean_batch.add_argument("--input-dir", required=True)
    clean_batch.add_argument("--pattern", default="*.csv")
    clean_batch.add_argument("--out-dir", default="data/cleaned_batch")
    clean_batch.add_argument("--name", default="合并数据集")
    clean_batch.add_argument("--freq", default="B")
    clean_batch.add_argument("--fill", choices=["ffill", "interpolate", "none"], default="ffill")
    clean_batch.add_argument("--no-derived-features", action="store_true")

    init_dict = sub.add_parser("init-dictionary", help="Create a data dictionary CSV from a prepared dataset.")
    init_dict.add_argument("--data", required=True)
    init_dict.add_argument("--out", default="data/data_dictionary.csv")
    init_dict.add_argument("--raw", default=None, help="Optional original legacy CSV for codes/units/source.")

    check_dict = sub.add_parser("validate-dictionary", help="Validate a data dictionary against a dataset.")
    check_dict.add_argument("--data", required=True)
    check_dict.add_argument("--dictionary", required=True)
    check_dict.add_argument("--report", default="reports/data_dictionary_validation.json")

    dm_search = sub.add_parser("dm-search-levels", help="Search local DM EDB level dictionary.")
    dm_search.add_argument("--keyword", required=True)
    dm_search.add_argument("--levels", default="references/dm_api/edb_indicator_levels.csv")
    dm_search.add_argument("--out", default=None)
    dm_search.add_argument("--max-rows", type=int, default=50)

    dm_creds = sub.add_parser("dm-credentials-help", help="Show how to provide DM credentials safely.")

    dm_check = sub.add_parser("dm-check", help="Check local DM SDK dependencies and credential environment variables.")
    dm_check.add_argument("--wheel", default="references/dm_api/dm_quant_api_client-0.2.3-py3-none-any.whl")

    dm_codes = sub.add_parser("dm-fetch-codes", help="Fetch DM EDB indicator ids by EDB level id.")
    dm_codes.add_argument("--level-ids", required=True, help="Comma-separated edb_level_id values.")
    dm_codes.add_argument("--out", required=True)
    dm_codes.add_argument("--base-url", default=None)
    dm_codes.add_argument("--timeout", type=int, default=30)
    dm_codes.add_argument("--max-pages", type=int, default=100)

    dm_codes_config = sub.add_parser("dm-fetch-codes-config", help="Fetch DM EDB indicator ids from a level config CSV.")
    dm_codes_config.add_argument("--config", required=True)
    dm_codes_config.add_argument("--out", required=True)
    dm_codes_config.add_argument("--base-url", default=None)
    dm_codes_config.add_argument("--timeout", type=int, default=30)
    dm_codes_config.add_argument("--max-pages", type=int, default=100)

    dm_data = sub.add_parser("dm-fetch-data", help="Fetch DM EDB indicator values by indicator id.")
    dm_data.add_argument("--indicator-id", required=True)
    dm_data.add_argument("--start-date", required=True)
    dm_data.add_argument("--end-date", required=True)
    dm_data.add_argument("--frequency", default=None)
    dm_data.add_argument("--out", required=True)
    dm_data.add_argument("--base-url", default=None)
    dm_data.add_argument("--timeout", type=int, default=30)

    dm_data_batch = sub.add_parser("dm-fetch-data-batch", help="Fetch many DM EDB indicators from a CSV config.")
    dm_data_batch.add_argument("--config", required=True)
    dm_data_batch.add_argument("--out-dir", required=True)
    dm_data_batch.add_argument("--start-date", default=None)
    dm_data_batch.add_argument("--end-date", default=None)
    dm_data_batch.add_argument("--base-url", default=None)
    dm_data_batch.add_argument("--timeout", type=int, default=30)

    dm_yield_curve = sub.add_parser("dm-fetch-yield-curve", help="Fetch DM bond yield-curve function data.")
    dm_yield_curve.add_argument("--data-source", default="18", help="18=中债, 19=中证")
    dm_yield_curve.add_argument("--curve-name", required=True)
    dm_yield_curve.add_argument("--curve-terms", required=True, help="Comma-separated term list, max 5 per request.")
    dm_yield_curve.add_argument("--curve-type", default="1", help="1=到期, 2=即期, 3/4=远期")
    dm_yield_curve.add_argument("--start-date", required=True)
    dm_yield_curve.add_argument("--end-date", required=True)
    dm_yield_curve.add_argument("--out", required=True)
    dm_yield_curve.add_argument("--base-url", default=None)
    dm_yield_curve.add_argument("--timeout", type=int, default=30)

    dm_coverage = sub.add_parser("dm-coverage-report", help="Summarize actual local DM raw data coverage by indicator.")
    dm_coverage.add_argument("--config", required=True)
    dm_coverage.add_argument("--raw-dir", required=True)
    dm_coverage.add_argument("--out-dir", default="data/dm_coverage_report")
    dm_coverage.add_argument("--model-features", default=None)

    dm_curve_audit = sub.add_parser("dm-curve-history-audit", help="Audit actual DM API history length for alternative bond curve definitions.")
    dm_curve_audit.add_argument("--candidates", required=True)
    dm_curve_audit.add_argument("--out-dir", default="data/dm_api_length_audit/alternative_curve_history")
    dm_curve_audit.add_argument("--start-date", default="2011-01-01")
    dm_curve_audit.add_argument("--end-date", default="2026-06-24")
    dm_curve_audit.add_argument("--zhongzhai-codes", default=None)
    dm_curve_audit.add_argument("--offset", type=int, default=0)
    dm_curve_audit.add_argument("--limit", type=int, default=None)
    dm_curve_audit.add_argument("--refresh", action="store_true")
    dm_curve_audit.add_argument("--timeout", type=int, default=60)

    dm_wide = sub.add_parser("dm-edb-to-wide", help="Convert raw DM EDB rows to point-in-time wide data.")
    dm_wide.add_argument("--input", required=True, help="Raw EDB csv file or directory.")
    dm_wide.add_argument("--pattern", default="*.csv")
    dm_wide.add_argument("--out", required=True)
    dm_wide.add_argument("--dictionary-out", default=None)
    dm_wide.add_argument("--model-ready-out", default=None)
    dm_wide.add_argument("--freq", default="B")
    dm_wide.add_argument("--start-date", default=None)
    dm_wide.add_argument("--end-date", default=None)
    dm_wide.add_argument("--fallback-release-lag-days", type=int, default=10)
    dm_wide.add_argument("--model-ready-max-missing-ratio", type=float, default=0.2)

    train = sub.add_parser("train", help="Train and backtest one neural model.")
    add_experiment_args(train)
    train.add_argument("--model", choices=MODEL_CHOICES, default="lstm")
    train.add_argument("--out-dir", default="models/creditbond_model")

    compare = sub.add_parser("compare", help="Train several neural models and write a comparison table.")
    add_experiment_args(compare)
    compare.add_argument("--models", default="lstm,gru,tcn,transformer")
    compare.add_argument("--out-dir", default="models/model_compare")

    pred = sub.add_parser("predict", help="Predict the latest horizon from a saved model.")
    pred.add_argument("--model-dir", required=True)
    pred.add_argument("--data", required=True)
    pred.add_argument("--out", default=None)
    pred.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    importance = sub.add_parser("feature-importance", help="Measure model feature importance by permutation.")
    importance.add_argument("--model-dir", required=True)
    importance.add_argument("--data", required=True)
    importance.add_argument("--out", required=True)
    importance.add_argument("--eval-tail-ratio", type=float, default=0.15)
    importance.add_argument("--repeats", type=int, default=3)
    importance.add_argument("--seed", type=int, default=42)
    importance.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    select_features = sub.add_parser("select-features", help="Select a reusable feature list from importance results.")
    select_features.add_argument("--importance", required=True)
    select_features.add_argument("--out", required=True)
    select_features.add_argument("--top-n", type=int, default=40)
    select_features.add_argument("--min-drop", type=float, default=0.0)
    select_features.add_argument("--include-contains", default="", help="Comma-separated substrings to force include.")

    auto = sub.add_parser("auto-research", help="Run training, feature importance, feature selection, retraining, and report writing.")
    add_experiment_args(auto)
    auto.add_argument("--models", default="gru,tcn,transformer")
    auto.add_argument("--out-dir", default="models/auto_research")
    auto.add_argument("--first-top-n", type=int, default=40)
    auto.add_argument("--second-top-n", type=int, default=20)
    auto.add_argument("--first-min-drop", type=float, default=0.0)
    auto.add_argument("--second-min-drop", type=float, default=0.0)
    auto.add_argument("--include-contains", default="", help="Comma-separated substrings to force include in the first selection.")
    auto.add_argument("--importance-repeats", type=int, default=3)
    auto.add_argument("--importance-eval-tail-ratio", type=float, default=0.15)
    auto.add_argument("--skip-second-selection", action="store_true")

    daily = sub.add_parser("daily-dm-update", help="Update all configured DM data, build model features, and optionally run latest predictions.")
    daily.add_argument("--config", required=True)
    daily.add_argument("--out-dir", default="data/dm_daily")
    daily.add_argument("--raw-dir", default=None)
    daily.add_argument("--processed-dir", default=None)
    daily.add_argument("--report-dir", default=None)
    daily.add_argument("--start-date", default=None)
    daily.add_argument("--end-date", default=None)
    daily.add_argument("--freq", default="B")
    daily.add_argument("--fallback-release-lag-days", type=int, default=10)
    daily.add_argument("--model-ready-max-missing-ratio", type=float, default=0.2)
    daily.add_argument("--no-derived-features", action="store_true")
    daily.add_argument("--skip-fetch", action="store_true")
    daily.add_argument("--full-refresh", action="store_true")
    daily.add_argument("--overlap-days", type=int, default=15)
    daily.add_argument("--model-dir", action="append", default=[])
    daily.add_argument("--models-root", default=None)
    daily.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    daily.add_argument("--base-url", default=None)
    daily.add_argument("--timeout", type=int, default=30)
    daily.add_argument("--strict-predictions", action="store_true")

    position_bt = sub.add_parser("position-backtest", help="Backtest a simple tenor-level position rule from saved model test predictions.")
    position_bt.add_argument("--features", required=True)
    position_bt.add_argument("--model-dir", action="append", required=True)
    position_bt.add_argument("--out-dir", default="data/backtests/position_backtest")
    position_bt.add_argument("--initial-position", type=float, default=0.5)
    position_bt.add_argument("--step", type=float, default=0.05)
    position_bt.add_argument("--transaction-cost-bp", type=float, default=0.0)
    position_bt.add_argument("--annual-days", type=int, default=252)
    position_bt.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    position_bt.add_argument("--signal-scope", choices=["test", "all"], default="test")
    position_bt.add_argument("--no-carry", action="store_true")

    strategy_sweep = sub.add_parser("no-data-strategy-sweep", help="Sweep probability thresholds, position steps, and model ensembles without adding data.")
    strategy_sweep.add_argument("--features", required=True)
    strategy_sweep.add_argument("--model-dir", action="append", required=True)
    strategy_sweep.add_argument("--out-dir", default="data/experiments/no_data_strategy_sweep")
    strategy_sweep.add_argument("--prob-thresholds", default="0.0,0.35,0.4,0.45,0.5,0.55,0.6")
    strategy_sweep.add_argument("--margins", default="0.0,0.03,0.06")
    strategy_sweep.add_argument("--steps", default="0.02,0.05,0.1")
    strategy_sweep.add_argument("--signal-scope", choices=["test", "all"], default="all")
    strategy_sweep.add_argument("--initial-position", type=float, default=0.5)
    strategy_sweep.add_argument("--transaction-cost-bp", type=float, default=0.0)
    strategy_sweep.add_argument("--annual-days", type=int, default=252)
    strategy_sweep.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    strategy_sweep.add_argument("--no-carry", action="store_true")
    strategy_sweep.add_argument("--no-ensembles", action="store_true")

    label_sweep = sub.add_parser("label-threshold-sweep", help="Retrain models over several label thresholds without adding data.")
    label_sweep.add_argument("--data", required=True)
    label_sweep.add_argument("--target-col", action="append", required=True)
    label_sweep.add_argument("--out-dir", default="models/no_data_label_threshold_sweep")
    label_sweep.add_argument("--theta-quantiles", default="0.45,0.55,0.65")
    label_sweep.add_argument("--models", default="transformer")
    label_sweep.add_argument("--horizon", type=int, default=5)
    label_sweep.add_argument("--window", type=int, default=60)
    label_sweep.add_argument("--train-ratio", type=float, default=0.7)
    label_sweep.add_argument("--val-ratio", type=float, default=0.15)
    label_sweep.add_argument("--train-end", default=None)
    label_sweep.add_argument("--val-end", default=None)
    label_sweep.add_argument("--include-target-feature", action="store_true")
    label_sweep.add_argument("--derive-features", action="store_true")
    label_sweep.add_argument("--epochs", type=int, default=25)
    label_sweep.add_argument("--batch-size", type=int, default=128)
    label_sweep.add_argument("--lr", type=float, default=1e-3)
    label_sweep.add_argument("--patience", type=int, default=5)
    label_sweep.add_argument("--hidden-size", type=int, default=64)
    label_sweep.add_argument("--layers", type=int, default=2)
    label_sweep.add_argument("--dropout", type=float, default=0.2)
    label_sweep.add_argument("--heads", type=int, default=4)
    label_sweep.add_argument("--kernel-size", type=int, default=3)
    label_sweep.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    label_sweep.add_argument("--no-class-weight", action="store_true")
    label_sweep.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    label_sweep.add_argument("--yield-unit", choices=["percent", "bp"], default="percent")
    label_sweep.add_argument("--seed", type=int, default=42)

    rolling = sub.add_parser("rolling-validation", help="Run expanding-window rolling validation without adding data.")
    rolling.add_argument("--data", required=True)
    rolling.add_argument("--target-col", action="append", required=True)
    rolling.add_argument("--out-dir", default="models/no_data_rolling_validation")
    rolling.add_argument("--models", default="transformer")
    rolling.add_argument("--theta-quantile", type=float, default=0.6)
    rolling.add_argument("--horizon", type=int, default=5)
    rolling.add_argument("--window", type=int, default=60)
    rolling.add_argument("--min-train-rows", type=int, default=180)
    rolling.add_argument("--val-rows", type=int, default=45)
    rolling.add_argument("--test-rows", type=int, default=45)
    rolling.add_argument("--step-rows", type=int, default=45)
    rolling.add_argument("--max-folds", type=int, default=3)
    rolling.add_argument("--include-target-feature", action="store_true")
    rolling.add_argument("--derive-features", action="store_true")
    rolling.add_argument("--epochs", type=int, default=20)
    rolling.add_argument("--batch-size", type=int, default=128)
    rolling.add_argument("--lr", type=float, default=1e-3)
    rolling.add_argument("--patience", type=int, default=4)
    rolling.add_argument("--hidden-size", type=int, default=64)
    rolling.add_argument("--layers", type=int, default=2)
    rolling.add_argument("--dropout", type=float, default=0.2)
    rolling.add_argument("--heads", type=int, default=4)
    rolling.add_argument("--kernel-size", type=int, default=3)
    rolling.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    rolling.add_argument("--no-class-weight", action="store_true")
    rolling.add_argument("--duration-map", default="3=2.8,5=4.5,10=7.5,20=12.0")
    rolling.add_argument("--yield-unit", choices=["percent", "bp"], default="percent")
    rolling.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "make-demo":
        df = make_demo_dataset(args.out, days=args.days, seed=args.seed)
        print(f"wrote {args.out} with shape {df.shape}")
        return 0

    if args.command == "prepare-wide":
        df = prepare_wide_data(args.input, args.out, freq=args.freq, fill=args.fill, derive_features=args.derive_features)
        print(f"wrote {args.out} with shape {df.shape}")
        return 0

    if args.command == "merge-exports":
        df = merge_exported_files(args.input_dir, args.out, args.mapping, freq=args.freq, fill=args.fill)
        print(f"wrote {args.out} with shape {df.shape}")
        return 0

    if args.command == "clean-data":
        report = clean_dataset(
            input_path=args.input,
            output_dir=args.out_dir,
            name=args.name,
            freq=args.freq,
            fill=args.fill,
            derive_features=not args.no_derived_features,
            raw_metadata_path=args.raw_metadata,
        )
        print(json.dumps(report["输出文件"], ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "清洗后形状": report["清洗后形状"],
                    "增强特征形状": report["增强特征形状"],
                    "清洗后日期范围": report["清洗后日期范围"],
                    "禁止向后填充": report["禁止向后填充"],
                    "数据字典校验通过": report["数据字典校验通过"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "clean-batch":
        report = clean_data_folder(
            input_dir=args.input_dir,
            output_dir=args.out_dir,
            pattern=args.pattern,
            name=args.name,
            freq=args.freq,
            fill=args.fill,
            derive_features=not args.no_derived_features,
        )
        print(json.dumps(report["输出文件"], ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "文件数量": report["文件数量"],
                    "历史合并形状": report["历史合并形状"],
                    "模型就绪形状": report["模型就绪形状"],
                    "增强特征形状": report["增强特征形状"],
                    "模型就绪日期范围": report["模型就绪日期范围"],
                    "禁止向后填充": report["禁止向后填充"],
                    "数据字典校验通过": report["数据字典校验通过"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "init-dictionary":
        dictionary = create_data_dictionary(args.data, args.out, raw_path=args.raw)
        print(f"wrote {args.out} with {len(dictionary)} indicators")
        return 0

    if args.command == "validate-dictionary":
        report = validate_data_dictionary(args.data, args.dictionary, report_path=args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["是否通过"] else 2

    if args.command == "dm-search-levels":
        result = search_edb_levels(args.keyword, path=args.levels, max_rows=args.max_rows)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"wrote {out} with {len(result)} rows")
        print(result.to_string(index=False))
        return 0

    if args.command == "dm-credentials-help":
        print(credentials_help())
        return 0

    if args.command == "dm-check":
        print(json.dumps(check_dm_environment(args.wheel), ensure_ascii=False, indent=2))
        return 0

    if args.command == "dm-fetch-codes":
        level_ids = [item.strip() for item in args.level_ids.split(",") if item.strip()]
        result = fetch_edb_indicator_codes(
            edb_level_ids=level_ids,
            base_url=args.base_url,
            timeout=args.timeout,
            max_pages=args.max_pages,
        )
        save_dataframe(result, args.out)
        print(f"wrote {args.out} with {len(result)} rows")
        return 0

    if args.command == "dm-fetch-codes-config":
        config = pd.read_csv(args.config, encoding="utf-8-sig")
        if "edb_level_id" not in config.columns:
            raise ValueError("配置文件必须包含 edb_level_id 列。")
        level_ids = [str(item).strip() for item in config["edb_level_id"].dropna().tolist() if str(item).strip()]
        result = fetch_edb_indicator_codes(
            edb_level_ids=level_ids,
            base_url=args.base_url,
            timeout=args.timeout,
            max_pages=args.max_pages,
        )
        save_dataframe(result, args.out)
        print(f"wrote {args.out} with {len(result)} rows")
        return 0

    if args.command == "dm-fetch-data":
        result = fetch_edb_indicator_data(
            indicator_id=args.indicator_id,
            start_date=args.start_date,
            end_date=args.end_date,
            frequency=args.frequency,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        save_dataframe(result, args.out)
        print(f"wrote {args.out} with {len(result)} rows")
        return 0

    if args.command == "dm-fetch-data-batch":
        result = fetch_edb_data_from_config(
            config_path=args.config,
            output_dir=args.out_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dm-fetch-yield-curve":
        terms = [item.strip() for item in args.curve_terms.split(",") if item.strip()]
        result = fetch_bond_yield_curve_data(
            data_source=args.data_source,
            curve_name=args.curve_name,
            curve_terms=terms,
            curve_type=args.curve_type,
            start_date=args.start_date,
            end_date=args.end_date,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        save_dataframe(result, args.out)
        print(f"wrote {args.out} with {len(result)} rows")
        return 0

    if args.command == "dm-coverage-report":
        result = run_dm_coverage_report(
            config_path=args.config,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            model_features_path=args.model_features,
        )
        print(json.dumps(
            {
                "HTML报告": result["html_report"],
                "明细CSV": result["detail_csv"],
                "汇总JSON": result["summary_json"],
                "指标数量": result["indicator_count"],
                "模型特征表": result["model_info"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if args.command == "dm-curve-history-audit":
        result = run_curve_history_audit(
            candidates_path=args.candidates,
            out_dir=args.out_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            zhongzhai_codes_path=args.zhongzhai_codes,
            offset=args.offset,
            limit=args.limit,
            refresh=args.refresh,
            timeout=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dm-edb-to-wide":
        result = edb_raw_to_point_in_time_wide(
            input_path=args.input,
            output_path=args.out,
            dictionary_path=args.dictionary_out,
            model_ready_path=args.model_ready_out,
            pattern=args.pattern,
            freq=args.freq,
            start_date=args.start_date,
            end_date=args.end_date,
            fallback_release_lag_days=args.fallback_release_lag_days,
            model_ready_max_missing_ratio=args.model_ready_max_missing_ratio,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train":
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset = build_dataset_from_args(args)
        print(
            "dataset:",
            {
                "train": dataset.X_train.shape,
                "val": dataset.X_val.shape,
                "test": dataset.X_test.shape,
                "features": len(dataset.feature_cols),
                "theta": dataset.theta,
            },
        )
        model_config = model_config_from_args(args, len(dataset.feature_cols), args.model)
        train_config = train_config_from_args(args)
        metrics = train_model(dataset, args.out_dir, model_config, train_config)
        print(json.dumps(metrics["backtest"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare":
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        requested_models = parse_model_list(args.models)
        run_model_comparison(args, args.out_dir, requested_models, features_file=args.features_file)
        return 0

    if args.command == "predict":
        result = predict_latest(args.model_dir, args.data, out_path=args.out, device_name=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "feature-importance":
        result = model_feature_importance(
            model_dir=args.model_dir,
            data_path=args.data,
            out_path=args.out,
            eval_tail_ratio=args.eval_tail_ratio,
            repeats=args.repeats,
            seed=args.seed,
            device_name=args.device,
        )
        print(result.head(30).to_string(index=False))
        print(f"wrote {args.out} with {len(result)} rows")
        return 0

    if args.command == "select-features":
        tokens = [item.strip() for item in args.include_contains.split(",") if item.strip()]
        result = select_features_from_importance(
            importance_path=args.importance,
            out_path=args.out,
            top_n=args.top_n,
            min_drop=args.min_drop,
            include_contains=tokens,
        )
        print(result[["feature", "importance_drop"]].head(50).to_string(index=False))
        print(f"wrote {args.out} with {len(result)} features")
        return 0

    if args.command == "auto-research":
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        requested_models = parse_model_list(args.models)
        base_out = Path(args.out_dir)
        base_out.mkdir(parents=True, exist_ok=True)

        print("\n===== step 1: full feature comparison =====")
        full_dir = base_out / "01_full_features"
        full_comparison = run_model_comparison(args, full_dir, requested_models, features_file=args.features_file)
        full_best_dir = best_model_dir(full_comparison)

        print("\n===== step 2: feature importance from best full model =====")
        first_importance_path = base_out / "02_full_model_feature_importance.csv"
        model_feature_importance(
            model_dir=full_best_dir,
            data_path=args.data,
            out_path=first_importance_path,
            eval_tail_ratio=args.importance_eval_tail_ratio,
            repeats=args.importance_repeats,
            seed=args.seed,
            device_name=args.device,
        )

        print("\n===== step 3: first feature selection =====")
        first_features_path = base_out / f"03_selected_top{args.first_top_n}.txt"
        tokens = [item.strip() for item in args.include_contains.split(",") if item.strip()]
        first_selected = select_features_from_importance(
            importance_path=first_importance_path,
            out_path=first_features_path,
            top_n=args.first_top_n,
            min_drop=args.first_min_drop,
            include_contains=tokens,
        )
        print(first_selected[["feature", "importance_drop"]].head(50).to_string(index=False))
        print(f"wrote {first_features_path} with {len(first_selected)} features")

        print("\n===== step 4: selected feature comparison =====")
        selected_dir = base_out / "04_selected_features"
        selected_comparison = run_model_comparison(args, selected_dir, requested_models, features_file=first_features_path)

        positive_comparison = None
        second_importance_path = None
        second_features_path = None
        if not args.skip_second_selection:
            selected_best_dir = best_model_dir(selected_comparison)
            print("\n===== step 5: feature importance from best selected model =====")
            second_importance_path = base_out / "05_selected_model_feature_importance.csv"
            model_feature_importance(
                model_dir=selected_best_dir,
                data_path=args.data,
                out_path=second_importance_path,
                eval_tail_ratio=args.importance_eval_tail_ratio,
                repeats=args.importance_repeats,
                seed=args.seed,
                device_name=args.device,
            )

            print("\n===== step 6: second feature selection =====")
            second_features_path = base_out / f"06_positive_top{args.second_top_n}.txt"
            second_selected = select_features_from_importance(
                importance_path=second_importance_path,
                out_path=second_features_path,
                top_n=args.second_top_n,
                min_drop=args.second_min_drop,
                include_contains=[],
            )
            print(second_selected[["feature", "importance_drop"]].head(50).to_string(index=False))
            print(f"wrote {second_features_path} with {len(second_selected)} features")

            print("\n===== step 7: positive feature comparison =====")
            positive_dir = base_out / "07_positive_features"
            positive_comparison = run_model_comparison(args, positive_dir, requested_models, features_file=second_features_path)

        report_path = base_out / "auto_research_report.md"
        write_auto_research_report(
            out_path=report_path,
            target_col=args.target_col,
            horizon=args.horizon,
            window=args.window,
            full_comparison=full_comparison,
            selected_comparison=selected_comparison,
            positive_comparison=positive_comparison,
            first_features_path=first_features_path,
            second_features_path=second_features_path,
            first_importance_path=first_importance_path,
            second_importance_path=second_importance_path,
        )
        print(f"report saved to {report_path}")
        return 0

    if args.command == "daily-dm-update":
        report = run_daily_dm_update(
            config_path=args.config,
            out_dir=args.out_dir,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            report_dir=args.report_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            freq=args.freq,
            fallback_release_lag_days=args.fallback_release_lag_days,
            model_ready_max_missing_ratio=args.model_ready_max_missing_ratio,
            derive_features=not args.no_derived_features,
            skip_fetch=args.skip_fetch,
            incremental=not args.full_refresh,
            overlap_days=args.overlap_days,
            model_dirs=args.model_dir,
            models_root=args.models_root,
            device_name=args.device,
            base_url=args.base_url,
            timeout=args.timeout,
            strict_predictions=args.strict_predictions,
        )
        print(json.dumps(
            {
                "特征文件": report["features_path"],
                "特征表形状": report["features_shape"],
                "数据日期范围": [report["data_start"], report["data_end"]],
                "预测数量": len(report["predictions"]),
                "HTML报告": report["html_report_path"],
                "Markdown报告": report["markdown_report_path"],
                "JSON报告": report["json_report_path"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if args.command == "position-backtest":
        report = run_position_backtest(
            PositionBacktestConfig(
                features_path=args.features,
                model_dirs=args.model_dir,
                out_dir=args.out_dir,
                initial_position=args.initial_position,
                step=args.step,
                transaction_cost_bp=args.transaction_cost_bp,
                annual_days=args.annual_days,
                include_carry=not args.no_carry,
                duration_map=parse_duration_map(args.duration_map),
                signal_scope=args.signal_scope,
            )
        )
        print(json.dumps(
            {
                "HTML报告": report["html_path"],
                "JSON报告": report["json_path"],
                "汇总CSV": report["summary_csv"],
                "模型数量": len(report["summary"]),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if args.command == "no-data-strategy-sweep":
        result = run_strategy_sweep(
            StrategySweepConfig(
                features_path=args.features,
                model_dirs=args.model_dir,
                out_dir=args.out_dir,
                prob_thresholds=parse_float_list(args.prob_thresholds, [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]),
                margins=parse_float_list(args.margins, [0.0, 0.03, 0.06]),
                steps=parse_float_list(args.steps, [0.02, 0.05, 0.1]),
                signal_scope=args.signal_scope,
                initial_position=args.initial_position,
                transaction_cost_bp=args.transaction_cost_bp,
                annual_days=args.annual_days,
                include_carry=not args.no_carry,
                include_ensembles=not args.no_ensembles,
                duration_map=parse_duration_map(args.duration_map),
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "label-threshold-sweep":
        result = run_label_threshold_sweep(
            LabelSweepConfig(
                data_path=args.data,
                target_cols=decode_cli_text_list(args.target_col),
                out_dir=args.out_dir,
                theta_quantiles=parse_float_list(args.theta_quantiles, [0.45, 0.55, 0.65]),
                models=parse_str_list(args.models, ["transformer"]),
                horizon=args.horizon,
                window=args.window,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                train_end=args.train_end,
                val_end=args.val_end,
                exclude_target_feature=not args.include_target_feature,
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
                class_weight=not args.no_class_weight,
                duration_map=parse_duration_map(args.duration_map),
                yield_unit=args.yield_unit,
                seed=args.seed,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "rolling-validation":
        result = run_rolling_validation(
            RollingValidationConfig(
                data_path=args.data,
                target_cols=decode_cli_text_list(args.target_col),
                out_dir=args.out_dir,
                models=parse_str_list(args.models, ["transformer"]),
                theta_quantile=args.theta_quantile,
                horizon=args.horizon,
                window=args.window,
                min_train_rows=args.min_train_rows,
                val_rows=args.val_rows,
                test_rows=args.test_rows,
                step_rows=args.step_rows,
                max_folds=args.max_folds,
                exclude_target_feature=not args.include_target_feature,
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
                class_weight=not args.no_class_weight,
                duration_map=parse_duration_map(args.duration_map),
                yield_unit=args.yield_unit,
                seed=args.seed,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
