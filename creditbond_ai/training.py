from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowedDataset, save_scaler
from .labels import LABEL_NAMES, LABEL_NAMES_CN, label_to_signal
from .model import ModelConfig, build_model, model_config_dict


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 8
    device: str = "auto"
    class_weight: bool = True
    duration: float = 3.0
    yield_unit: str = "percent"


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    return torch.device(requested)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.ones(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    probs = []
    for batch in loader:
        logits = model(batch.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


@torch.no_grad()
def evaluate_loss_f1(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        losses.append(float(loss_fn(logits, yb).detach().cpu()))
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(np.mean(losses)), float(macro_f1)


def train_model(
    dataset: WindowedDataset,
    out_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(train_config.device)

    model = build_model(model_config_dict(model_config)).to(device)
    train_loader = make_loader(dataset.X_train, dataset.y_train, train_config.batch_size, shuffle=True)
    val_loader = make_loader(dataset.X_val, dataset.y_val, train_config.batch_size, shuffle=False)

    weight = class_weights(dataset.y_train).to(device) if train_config.class_weight else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=1e-4)

    history = []
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_f1 = evaluate_loss_f1(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_config.patience:
                print(f"early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_assets(model, dataset, out_dir, model_config, train_config, history, best_f1)
    metrics = evaluate_and_save(model, dataset, out_dir, train_config, device)
    plot_history(history, out_dir)
    return metrics


def save_assets(
    model: nn.Module,
    dataset: WindowedDataset,
    out_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    history: list[dict],
    best_f1: float,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_config_dict(model_config),
        "train_config": asdict(train_config),
        "feature_cols": dataset.feature_cols,
        "target_col": dataset.target_col,
        "horizon": dataset.horizon,
        "window": dataset.window,
        "theta": dataset.theta,
        "label_names": LABEL_NAMES,
        "label_names_cn": LABEL_NAMES_CN,
        "best_val_macro_f1": best_f1,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    save_scaler(dataset.scaler, out_dir / "scaler.joblib")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "feature_cols.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(dataset.feature_cols))


def evaluate_and_save(
    model: nn.Module,
    dataset: WindowedDataset,
    out_dir: Path,
    train_config: TrainConfig,
    device: torch.device,
) -> dict:
    probs = predict_proba(model, dataset.X_test, device=device)
    pred = probs.argmax(axis=1)
    report = classification_report(
        dataset.y_test,
        pred,
        labels=[0, 1, 2],
        target_names=[LABEL_NAMES_CN[i] for i in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(dataset.y_test, pred, labels=[0, 1, 2])
    backtest = backtest_directional_proxy(
        future_changes=dataset.changes_test,
        pred=pred,
        duration=train_config.duration,
        yield_unit=train_config.yield_unit,
    )

    pred_df = pd.DataFrame(
        {
            "date": dataset.dates_test,
            "y_true": dataset.y_test,
            "y_true_cn": [LABEL_NAMES_CN[int(x)] for x in dataset.y_test],
            "y_pred": pred,
            "y_pred_cn": [LABEL_NAMES_CN[int(x)] for x in pred],
            "future_yield_change": dataset.changes_test,
            "prob_bearish": probs[:, 0],
            "prob_bullish": probs[:, 1],
            "prob_range": probs[:, 2],
            "pnl_proxy": backtest["pnl_proxy"],
            "cum_pnl_proxy": backtest["cum_pnl_proxy"],
        }
    )
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=["true_bearish", "true_bullish", "true_range"], columns=["pred_bearish", "pred_bullish", "pred_range"]).to_csv(
        out_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "backtest": {k: v for k, v in backtest.items() if not isinstance(v, np.ndarray)},
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    plot_backtest(pred_df, out_dir)
    return metrics


def backtest_directional_proxy(
    future_changes: np.ndarray,
    pred: np.ndarray,
    duration: float,
    yield_unit: str,
) -> dict:
    signals = np.asarray([label_to_signal(int(x)) for x in pred], dtype=float)
    if yield_unit == "percent":
        delta_yield_decimal = future_changes / 100.0
    elif yield_unit == "bp":
        delta_yield_decimal = future_changes / 10000.0
    else:
        raise ValueError("yield_unit must be percent or bp")

    pnl_proxy = -duration * signals * delta_yield_decimal
    cum_pnl_proxy = np.cumsum(pnl_proxy)
    active = signals != 0
    return {
        "total_return_proxy": float(cum_pnl_proxy[-1]) if len(cum_pnl_proxy) else 0.0,
        "mean_active_return_proxy": float(pnl_proxy[active].mean()) if active.any() else 0.0,
        "active_signal_ratio": float(active.mean()) if len(active) else 0.0,
        "positive_active_ratio": float((pnl_proxy[active] > 0).mean()) if active.any() else 0.0,
        "pnl_proxy": pnl_proxy,
        "cum_pnl_proxy": cum_pnl_proxy,
    }


def plot_history(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="validation macro F1")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=160)
    plt.close(fig)


def plot_backtest(pred_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(pred_df["date"]), pred_df["cum_pnl_proxy"])
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Directional Backtest Proxy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative proxy return")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_proxy.png", dpi=160)
    plt.close(fig)
