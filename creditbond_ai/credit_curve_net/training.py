from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import WindowedDataset, save_scaler
from ..training import backtest_directional_proxy, choose_device, class_weights, plot_backtest, plot_history
from .architecture import CreditCurveNet, CreditCurveNetConfig, credit_curve_net_config_dict


LABEL_NAMES = {0: "bearish", 1: "bullish", 2: "range"}
LABEL_NAMES_CN = {0: "\u770b\u7a7a", 1: "\u770b\u591a", 2: "\u9707\u8361"}


@dataclass
class CreditCurveTrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 8e-4
    patience: int = 12
    device: str = "auto"
    class_weight: bool = True
    duration: float = 3.0
    yield_unit: str = "percent"
    aux_change_weight: float = 0.15
    label_smoothing: float = 0.03
    weight_decay: float = 2e-4
    grad_clip: float = 2.0
    amp: bool = True


def _changes_to_bp(changes: np.ndarray, yield_unit: str) -> np.ndarray:
    if yield_unit == "percent":
        return (changes * 100.0).astype(np.float32)
    if yield_unit == "bp":
        return changes.astype(np.float32)
    raise ValueError("yield_unit must be percent or bp")


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    changes: np.ndarray,
    batch_size: int,
    shuffle: bool,
    yield_unit: str,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(_changes_to_bp(changes, yield_unit)).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _loss_from_outputs(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
    change_bp: torch.Tensor,
    class_loss_fn: nn.Module,
    change_loss_fn: nn.Module,
    aux_change_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = outputs["logits"]
    class_loss = class_loss_fn(logits, y)
    change_loss = change_loss_fn(outputs["change_bp"], change_bp)
    loss = class_loss + aux_change_weight * change_loss
    return loss, class_loss, change_loss


@torch.no_grad()
def _evaluate_loss_f1(
    model: CreditCurveNet,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    change_loss_fn: nn.Module,
    train_config: CreditCurveTrainConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb, change_bp in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        change_bp = change_bp.to(device)
        outputs = model(xb)
        loss, _, _ = _loss_from_outputs(
            outputs,
            yb,
            change_bp,
            class_loss_fn,
            change_loss_fn,
            train_config.aux_change_weight,
        )
        losses.append(float(loss.detach().cpu()))
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(outputs["logits"].argmax(dim=1).detach().cpu().numpy().tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return float(np.mean(losses)), float(macro_f1), float(accuracy)


@torch.no_grad()
def predict_credit_curve_outputs(
    model: CreditCurveNet,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    probs = []
    change_bp = []
    feature_weights = []
    scale_weights = []
    regime_weights = []
    time_weights = []
    for xb in loader:
        outputs = model(xb.to(device))
        probs.append(torch.softmax(outputs["logits"], dim=1).detach().cpu().numpy())
        change_bp.append(outputs["change_bp"].detach().cpu().numpy())
        feature_weights.append(outputs["feature_weights"].detach().cpu().numpy())
        scale_weights.append(outputs["scale_weights"].detach().cpu().numpy())
        regime_weights.append(outputs["regime_weights"].detach().cpu().numpy())
        time_weights.append(outputs["time_weights"].detach().cpu().numpy())
    return {
        "probs": np.vstack(probs),
        "change_bp": np.concatenate(change_bp),
        "feature_weights": np.vstack(feature_weights),
        "scale_weights": np.vstack(scale_weights),
        "regime_weights": np.vstack(regime_weights),
        "time_weights": np.vstack(time_weights),
    }


def train_credit_curve_net(
    dataset: WindowedDataset,
    out_dir: str | Path,
    model_config: CreditCurveNetConfig,
    train_config: CreditCurveTrainConfig,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(train_config.device)
    model = CreditCurveNet(model_config).to(device)

    train_loader = _make_loader(
        dataset.X_train,
        dataset.y_train,
        dataset.changes_train,
        train_config.batch_size,
        shuffle=True,
        yield_unit=train_config.yield_unit,
    )
    val_loader = _make_loader(
        dataset.X_val,
        dataset.y_val,
        dataset.changes_val,
        train_config.batch_size,
        shuffle=False,
        yield_unit=train_config.yield_unit,
    )

    weight = class_weights(dataset.y_train).to(device) if train_config.class_weight else None
    class_loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=train_config.label_smoothing)
    change_loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=3)
    use_amp = bool(train_config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_class_losses: list[float] = []
        train_change_losses: list[float] = []

        for xb, yb, change_bp in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            change_bp = change_bp.to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.amp.autocast("cuda", enabled=True) if use_amp and hasattr(torch, "amp") else nullcontext()
            with amp_context:
                outputs = model(xb)
                loss, class_loss, change_loss = _loss_from_outputs(
                    outputs,
                    yb,
                    change_bp,
                    class_loss_fn,
                    change_loss_fn,
                    train_config.aux_change_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))
            train_class_losses.append(float(class_loss.detach().cpu()))
            train_change_losses.append(float(change_loss.detach().cpu()))

        val_loss, val_f1, val_accuracy = _evaluate_loss_f1(
            model,
            val_loader,
            class_loss_fn,
            change_loss_fn,
            train_config,
            device,
        )
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_class_loss": float(np.mean(train_class_losses)),
            "train_change_loss": float(np.mean(train_change_losses)),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_accuracy,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f} val_acc={val_accuracy:.4f}",
            flush=True,
        )

        improved = val_f1 > best_f1 or (np.isclose(val_f1, best_f1) and val_loss < best_loss)
        if improved:
            best_f1 = val_f1
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_config.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    _save_assets(model, dataset, out_dir, model_config, train_config, history, best_f1)
    metrics = _evaluate_and_save(model, dataset, out_dir, train_config, device, best_f1)
    plot_history(history, out_dir)
    return metrics


def _save_assets(
    model: CreditCurveNet,
    dataset: WindowedDataset,
    out_dir: Path,
    model_config: CreditCurveNetConfig,
    train_config: CreditCurveTrainConfig,
    history: list[dict],
    best_f1: float,
) -> None:
    checkpoint = {
        "model_family": "credit_curve_net",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": credit_curve_net_config_dict(model_config),
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
    (out_dir / "model_readme.md").write_text(
        "\n".join(
            [
                "# CreditCurveNet",
                "",
                "Custom neural model for China credit-bond yield-curve direction research.",
                "",
                "- Multi-scale temporal convolution branches: short, middle, and long market memory.",
                "- Feature gate: learns which indicators matter in the current window.",
                "- Tenor embedding: keeps 3Y/5Y/10Y/20Y behavior separate.",
                "- Regime gate: blends several expert heads for different market states.",
                "- Auxiliary change head: jointly learns future yield-change magnitude in bp.",
            ]
        ),
        encoding="utf-8",
    )


def _evaluate_and_save(
    model: CreditCurveNet,
    dataset: WindowedDataset,
    out_dir: Path,
    train_config: CreditCurveTrainConfig,
    device: torch.device,
    best_f1: float,
) -> dict:
    outputs = predict_credit_curve_outputs(model, dataset.X_test, device=device, batch_size=train_config.batch_size)
    probs = outputs["probs"]
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
            "predicted_change_bp": outputs["change_bp"],
            "prob_bearish": probs[:, 0],
            "prob_bullish": probs[:, 1],
            "prob_range": probs[:, 2],
            "pnl_proxy": backtest["pnl_proxy"],
            "cum_pnl_proxy": backtest["cum_pnl_proxy"],
        }
    )
    for idx in range(outputs["regime_weights"].shape[1]):
        pred_df[f"regime_weight_{idx + 1}"] = outputs["regime_weights"][:, idx]
    for idx in range(outputs["scale_weights"].shape[1]):
        pred_df[f"scale_weight_{idx + 1}"] = outputs["scale_weights"][:, idx]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    cm_df = pd.DataFrame(
        cm,
        index=["true_bearish", "true_bullish", "true_range"],
        columns=["pred_bearish", "pred_bullish", "pred_range"],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    feature_gate = pd.DataFrame(
        {
            "feature": dataset.feature_cols,
            "mean_gate": outputs["feature_weights"].mean(axis=0),
            "std_gate": outputs["feature_weights"].std(axis=0),
        }
    ).sort_values("mean_gate", ascending=False)
    feature_gate.to_csv(out_dir / "feature_gate_importance.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "model_family": "credit_curve_net",
        "best_val_macro_f1": best_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "backtest": {k: v for k, v in backtest.items() if not isinstance(v, np.ndarray)},
        "feature_gate_top20": feature_gate.head(20).to_dict(orient="records"),
        "mean_regime_weights": outputs["regime_weights"].mean(axis=0).tolist(),
        "mean_scale_weights": outputs["scale_weights"].mean(axis=0).tolist(),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    plot_backtest(pred_df, out_dir)
    return metrics
