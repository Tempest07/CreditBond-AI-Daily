from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class CreditCurveNetConfig:
    input_size: int
    window: int
    hidden_size: int = 192
    regime_count: int = 4
    dropout: float = 0.15
    n_classes: int = 3
    conv_kernels: tuple[int, ...] = (3, 7, 15)
    tenor_index: int = 0
    max_tenors: int = 8


def infer_tenor_index(target_col: str) -> int:
    text = str(target_col)
    tenor_to_index = {
        "3\u5e74": 0,
        "5\u5e74": 1,
        "10\u5e74": 2,
        "20\u5e74": 3,
        "30\u5e74": 4,
    }
    for marker, index in tenor_to_index.items():
        if marker in text:
            return index
    return 0


def credit_curve_net_config_dict(config: CreditCurveNetConfig) -> dict:
    return asdict(config)


class CreditCurveNet(nn.Module):
    """A credit-curve-specific multi-scale and regime-gated classifier."""

    def __init__(self, config: CreditCurveNetConfig):
        super().__init__()
        self.config = config
        kernels = tuple(int(k) for k in config.conv_kernels)
        if not kernels:
            raise ValueError("conv_kernels must not be empty.")
        if config.hidden_size < 16:
            raise ValueError("hidden_size is too small.")
        if config.regime_count < 1:
            raise ValueError("regime_count must be >= 1.")

        stats_size = config.input_size * 3
        self.input_norm = nn.LayerNorm(config.input_size)
        self.feature_gate = nn.Sequential(
            nn.Linear(stats_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.input_size),
            nn.Sigmoid(),
        )
        self.scale_selector = nn.Sequential(
            nn.Linear(stats_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, len(kernels)),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        config.input_size,
                        config.hidden_size,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(config.hidden_size),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel in kernels
            ]
        )

        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.time_attention = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        self.tenor_embedding = nn.Embedding(config.max_tenors, config.hidden_size)
        self.context_norm = nn.LayerNorm(config.hidden_size)
        self.context_dropout = nn.Dropout(config.dropout)

        self.regime_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.regime_count),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_size, config.n_classes),
                )
                for _ in range(config.regime_count)
            ]
        )
        self.change_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                x[:, -1, :],
                x.mean(dim=1),
                x.std(dim=1, unbiased=False),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("CreditCurveNet expects input shape (batch, window, features).")
        x = self.input_norm(x)
        stats = self._stats(x)
        feature_weights = self.feature_gate(stats)
        x_gated = x * feature_weights.unsqueeze(1)

        branch_input = x_gated.transpose(1, 2)
        branch_contexts = torch.stack(
            [branch(branch_input).squeeze(-1) for branch in self.branches],
            dim=1,
        )
        scale_weights = torch.softmax(self.scale_selector(stats), dim=1)
        multi_scale_context = torch.sum(branch_contexts * scale_weights.unsqueeze(-1), dim=1)

        seq = self.input_projection(x_gated)
        time_scores = self.time_attention(seq).squeeze(-1)
        time_weights = torch.softmax(time_scores, dim=1)
        time_context = torch.sum(seq * time_weights.unsqueeze(-1), dim=1)

        tenor_id = max(0, min(int(self.config.tenor_index), self.config.max_tenors - 1))
        tenor_ids = torch.full((x.size(0),), tenor_id, dtype=torch.long, device=x.device)
        tenor_context = self.tenor_embedding(tenor_ids)

        context = self.context_norm(time_context + multi_scale_context + tenor_context)
        context = self.context_dropout(context)
        regime_weights = torch.softmax(self.regime_gate(context), dim=1)
        expert_logits = torch.stack([expert(context) for expert in self.experts], dim=1)
        logits = torch.sum(expert_logits * regime_weights.unsqueeze(-1), dim=1)
        change_bp = self.change_head(context).squeeze(-1)

        return {
            "logits": logits,
            "change_bp": change_bp,
            "feature_weights": feature_weights,
            "scale_weights": scale_weights,
            "time_weights": time_weights,
            "regime_weights": regime_weights,
        }
