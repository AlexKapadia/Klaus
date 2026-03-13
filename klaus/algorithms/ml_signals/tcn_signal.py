"""Temporal Convolutional Network signal (PyTorch)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.config.settings import PROJECT_ROOT
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class _CausalConv1d(nn.Module):
    """Causal convolution: pads only on the left so output doesn't see future."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class _TCNBlock(nn.Module):
    """Single TCN residual block."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


class _TCNNet(nn.Module):
    """Temporal Convolutional Network for sequence classification."""

    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last = out[:, :, -1]  # take last timestep
        return self.sigmoid(self.fc(last))


@register_algorithm
class TCNSignal(BaseAlgorithm):
    """TCN-based directional signal.

    Same feature set and training approach as LSTM but using
    dilated causal convolutions for potentially better long-range patterns.
    """

    name = "tcn_signal"
    supported_instruments = ["XAUUSD", "XTIUSD"]
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 300

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.num_channels = self.params.get("num_channels", [32, 32, 32])
        self.kernel_size = self.params.get("kernel_size", 3)
        self.seq_length = self.params.get("sequence_length", 60)
        self.dropout = self.params.get("dropout", 0.2)
        self.retrain_days = self.params.get("retrain_interval_days", 7)

        self._model: Optional[_TCNNet] = None
        self._last_train_time: Optional[datetime] = None
        self._feature_cols = ["returns", "rolling_volatility", "rsi", "macd_hist", "bb_pct", "zscore"]
        self._device = "cpu"
        self._norm_mean = None
        self._norm_std = None

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        if "rolling_volatility" not in df.columns:
            df["rolling_volatility"] = df["returns"].rolling(20).std()
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = (100 - (100 / (1 + rs))) / 100
        if "macd_hist" not in df.columns:
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            df["macd_hist"] = macd - macd.ewm(span=9).mean()
        if "bb_pct" not in df.columns:
            sma = df["close"].rolling(20).mean()
            std = df["close"].rolling(20).std()
            df["bb_pct"] = (df["close"] - (sma - 2 * std)) / (4 * std).replace(0, np.nan)
        if "zscore" not in df.columns:
            rm = df["close"].rolling(60).mean()
            rs = df["close"].rolling(60).std()
            df["zscore"] = (df["close"] - rm) / rs.replace(0, np.nan)
        return df

    def fit(self, data: pd.DataFrame) -> None:
        if not _HAS_TORCH:
            return

        df = self._prepare_features(data)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
        df = df.dropna(subset=self._feature_cols + ["target"])

        if len(df) < self.seq_length + 100:
            logger.warning(f"TCN: insufficient data ({len(df)} rows)")
            return

        features = df[self._feature_cols].values
        self._norm_mean = features.mean(axis=0)
        self._norm_std = features.std(axis=0) + 1e-8
        features = (features - self._norm_mean) / self._norm_std

        targets = df["target"].values

        # Make sequences
        X, y = [], []
        for i in range(self.seq_length, len(features)):
            X.append(features[i - self.seq_length : i])
            y.append(targets[i])
        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train = torch.FloatTensor(X[:split]).to(self._device)
        y_train = torch.FloatTensor(y[:split]).unsqueeze(1).to(self._device)

        self._model = _TCNNet(
            input_size=len(self._feature_cols),
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(self._device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        batch_size = 64

        self._model.train()
        for epoch in range(20):
            indices = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i : i + batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                optimiser.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
                n_batches += 1

        self._model.eval()
        self._last_train_time = datetime.utcnow()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), MODEL_DIR / "tcn_latest.pt")
        logger.info(f"TCN trained on {split} sequences")

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not _HAS_TORCH or self._model is None:
            return None

        df = self._prepare_features(data)
        df = df.dropna(subset=self._feature_cols)

        if len(df) < self.seq_length:
            return None

        features = df[self._feature_cols].tail(self.seq_length).values
        features = (features - self._norm_mean) / self._norm_std

        X = torch.FloatTensor(features).unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            prob_up = self._model(X).item()

        if prob_up > 0.6:
            direction = Direction.LONG
            strength = (prob_up - 0.5) * 2
        elif prob_up < 0.4:
            direction = Direction.SHORT
            strength = (0.5 - prob_up) * 2
        else:
            return None

        strength = np.clip(strength, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"prob_up": float(prob_up)},
        )
