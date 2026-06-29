from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import load_scaler, load_wide_dataset
from ..training import choose_device
from .architecture import CreditCurveNet, CreditCurveNetConfig


LABEL_NAMES_CN = {0: "\u770b\u7a7a", 1: "\u770b\u591a", 2: "\u9707\u8361"}


def load_credit_curve_net(model_dir: str | Path, device: torch.device) -> tuple[CreditCurveNet, dict]:
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "credit_curve_net":
        raise ValueError(f"Not a CreditCurveNet checkpoint: {model_dir}")
    config = CreditCurveNetConfig(**checkpoint["model_config"])
    model = CreditCurveNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_latest_credit_curve_net(
    model_dir: str | Path,
    data_path: str | Path,
    out_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict:
    device = choose_device(device_name)
    model_dir = Path(model_dir)
    model, checkpoint = load_credit_curve_net(model_dir, device)
    scaler = load_scaler(model_dir / "scaler.joblib")

    feature_cols = checkpoint["feature_cols"]
    window = int(checkpoint["window"])
    horizon = int(checkpoint["horizon"])
    df = load_wide_dataset(data_path).replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction data is missing feature columns: {missing}")
    if len(df) < window:
        raise ValueError(f"Need at least {window} rows, got {len(df)}.")

    X = df[feature_cols].astype(float).to_numpy()
    X_scaled = scaler.transform(X).astype(np.float32)
    latest = torch.from_numpy(X_scaled[-window:][None, :, :]).float().to(device)
    with torch.no_grad():
        outputs = model(latest)
        probs = torch.softmax(outputs["logits"], dim=1).detach().cpu().numpy()[0]
        feature_weights = outputs["feature_weights"].detach().cpu().numpy()[0]
        scale_weights = outputs["scale_weights"].detach().cpu().numpy()[0]
        regime_weights = outputs["regime_weights"].detach().cpu().numpy()[0]
        change_bp = float(outputs["change_bp"].detach().cpu().numpy()[0])

    pred_label = int(np.argmax(probs))
    last_date = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")
    top_features = sorted(
        [
            {"feature": feature, "gate": float(weight)}
            for feature, weight in zip(feature_cols, feature_weights, strict=True)
        ],
        key=lambda item: item["gate"],
        reverse=True,
    )[:20]

    result = {
        "prediction_date": last_date,
        "horizon_days": horizon,
        "target_col": checkpoint.get("target_col", ""),
        "prediction": LABEL_NAMES_CN[pred_label],
        "prediction_label": pred_label,
        "probabilities": {
            LABEL_NAMES_CN[0]: float(probs[0]),
            LABEL_NAMES_CN[1]: float(probs[1]),
            LABEL_NAMES_CN[2]: float(probs[2]),
        },
        "predicted_change_bp": change_bp,
        "regime_weights": [float(x) for x in regime_weights],
        "scale_weights": [float(x) for x in scale_weights],
        "top_features": top_features,
        "model_dir": str(model_dir),
    }

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            with out.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
        else:
            flat = {
                "prediction_date": result["prediction_date"],
                "horizon_days": result["horizon_days"],
                "target_col": result["target_col"],
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "prob_bearish": result["probabilities"][LABEL_NAMES_CN[0]],
                "prob_bullish": result["probabilities"][LABEL_NAMES_CN[1]],
                "prob_range": result["probabilities"][LABEL_NAMES_CN[2]],
                "predicted_change_bp": result["predicted_change_bp"],
                "model_dir": result["model_dir"],
            }
            for idx, value in enumerate(result["regime_weights"], start=1):
                flat[f"regime_weight_{idx}"] = value
            for idx, value in enumerate(result["scale_weights"], start=1):
                flat[f"scale_weight_{idx}"] = value
            pd.DataFrame([flat]).to_csv(out, index=False, encoding="utf-8-sig")
    return result
