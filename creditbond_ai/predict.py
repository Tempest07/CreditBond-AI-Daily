from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_scaler, load_wide_dataset
from .labels import ADVICE_CN, LABEL_NAMES_CN
from .model import build_model
from .training import choose_device


def load_checkpoint(model_dir: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_latest(
    model_dir: str | Path,
    data_path: str | Path,
    out_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict:
    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(model_dir, device)
    scaler = load_scaler(Path(model_dir) / "scaler.joblib")

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
        probs = torch.softmax(model(latest), dim=1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    last_date = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")

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
        "advice": ADVICE_CN[pred_label],
        "model_dir": str(Path(model_dir)),
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            with out.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
        else:
            pd.DataFrame([result]).to_csv(out, index=False, encoding="utf-8-sig")
    return result
