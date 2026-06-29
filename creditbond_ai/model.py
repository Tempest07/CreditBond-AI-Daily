from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_size: int
    architecture: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    n_classes: int = 3
    n_heads: int = 4
    tcn_kernel_size: int = 3
    max_window: int = 512


class RecurrentClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        recurrent_cls = nn.GRU if config.architecture == "gru" else nn.LSTM
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = recurrent_cls(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.encoder(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_classes),
        )

    def encode_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.encoder(x)
        scores = self.attention(output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(output * weights.unsqueeze(-1), dim=1)
        return context, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context, _ = self.encode_with_attention(x)
        return self.head(context)

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context, weights = self.encode_with_attention(x)
        return self.head(context), weights


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.norm(out + self.downsample(x))


class TCNClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers = []
        in_channels = config.input_size
        for i in range(config.num_layers):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=config.hidden_size,
                    kernel_size=config.tcn_kernel_size,
                    dilation=2**i,
                    dropout=config.dropout,
                )
            )
            in_channels = config.hidden_size
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.encoder(x).transpose(1, 2)
        return self.head(encoded[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int):
        super().__init__()
        positions = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_size % config.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads for transformer.")
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.position = PositionalEncoding(config.hidden_size, config.max_window)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x)
        encoded = self.encoder(x)
        return self.head(encoded[:, -1, :])


def build_model(config_dict: dict) -> nn.Module:
    config = ModelConfig(**config_dict)
    architecture = config.architecture.lower()
    if architecture in {"lstm", "gru"}:
        return RecurrentClassifier(config)
    if architecture in {"lstm_attention", "lstm-attention"}:
        config.architecture = "lstm_attention"
        return LSTMAttentionClassifier(config)
    if architecture == "tcn":
        return TCNClassifier(config)
    if architecture == "transformer":
        return TransformerClassifier(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def model_config_dict(config: ModelConfig) -> dict:
    return asdict(config)
