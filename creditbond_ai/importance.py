from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .data import add_labels, load_scaler, load_wide_dataset, make_windows
from .predict import load_checkpoint
from .training import choose_device, predict_proba


def model_feature_importance(
    model_dir: str | Path,
    data_path: str | Path,
    out_path: str | Path,
    eval_tail_ratio: float = 0.15,
    repeats: int = 3,
    seed: int = 42,
    device_name: str = "auto",
) -> pd.DataFrame:
    if not 0 < eval_tail_ratio < 1:
        raise ValueError("eval_tail_ratio 必须在 0 到 1 之间。")
    if repeats < 1:
        raise ValueError("repeats 必须大于等于 1。")

    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(model_dir, device)
    scaler = load_scaler(Path(model_dir) / "scaler.joblib")
    feature_cols = checkpoint["feature_cols"]
    target_col = checkpoint["target_col"]
    horizon = int(checkpoint["horizon"])
    window = int(checkpoint["window"])

    raw = load_wide_dataset(data_path).replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")
    missing = [col for col in feature_cols + [target_col] if col not in raw.columns]
    if missing:
        raise ValueError(f"数据缺少模型需要的列: {missing}")
    labeled, _ = add_labels(raw, target_col=target_col, horizon=horizon, theta_quantile=0.6)
    X_rows = labeled[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")
    aligned = labeled.loc[X_rows.index]
    X_scaled = scaler.transform(X_rows.to_numpy(dtype=np.float64)).astype(np.float32)
    X_all, y_all, dates_all, _ = make_windows(
        X_scaled=X_scaled,
        y=aligned["label"].to_numpy(dtype=np.int64),
        dates=aligned["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        future_changes=aligned["future_yield_change"].to_numpy(dtype=np.float64),
        window=window,
    )

    eval_count = max(30, int(len(X_all) * eval_tail_ratio))
    eval_count = min(eval_count, len(X_all))
    X_eval = X_all[-eval_count:].copy()
    y_eval = y_all[-eval_count:].copy()
    dates_eval = dates_all[-eval_count:]

    base_prob = predict_proba(model, X_eval, device=device)
    base_pred = base_prob.argmax(axis=1)
    base_macro_f1 = f1_score(y_eval, base_pred, average="macro", zero_division=0)

    rng = np.random.default_rng(seed)
    rows = []
    for feature_index, feature_name in enumerate(feature_cols):
        scores = []
        for _ in range(repeats):
            X_perm = X_eval.copy()
            order = rng.permutation(len(X_perm))
            X_perm[:, :, feature_index] = X_perm[order, :, feature_index]
            prob = predict_proba(model, X_perm, device=device)
            pred = prob.argmax(axis=1)
            scores.append(f1_score(y_eval, pred, average="macro", zero_division=0))
        mean_score = float(np.mean(scores))
        rows.append(
            {
                "feature": feature_name,
                "base_macro_f1": float(base_macro_f1),
                "permuted_macro_f1": mean_score,
                "importance_drop": float(base_macro_f1 - mean_score),
                "eval_start": str(dates_eval[0]),
                "eval_end": str(dates_eval[-1]),
                "eval_count": int(eval_count),
                "repeats": int(repeats),
            }
        )
    result = pd.DataFrame(rows).sort_values("importance_drop", ascending=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result
