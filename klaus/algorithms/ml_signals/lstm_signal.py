"""LSTM sequence model for directional prediction (PyTorch)."""

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


class _LSTMNet(nn.Module):
    """Simple LSTM classifier for direction prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.sigmoid(self.fc(last_hidden))


@register_algorithm
class LSTMSignal(BaseAlgorithm):
    """LSTM-based directional signal using PyTorch.

    Input: sequence of [returns, vol, RSI, MACD, bb_pct, zscore] over N bars.
    Output: probability of up move.
    """

    name = "lstm_signal"
    supported_instruments = ["XAUUSD", "XTIUSD"]
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 300

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.hidden_size = self.params.get("hidden_size", 64)
        self.num_layers = self.params.get("num_layers", 2)
        self.seq_length = self.params.get("sequence_length", 60)
        self.dropout = self.params.get("dropout", 0.2)
        self.retrain_days = self.params.get("retrain_interval_days", 7)

        self._model: Optional[_LSTMNet] = None
        self._last_train_time: Optional[datetime] = None
        self._feature_cols = ["returns", "rolling_volatility", "rsi", "macd_hist", "bb_pct", "zscore"]
        self._input_size = len(self._feature_cols)
        self._device = "cpu"

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
            df["rsi"] = (100 - (100 / (1 + rs))) / 100  # normalise to 0-1
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

    def _make_sequences(self, features: np.ndarray, targets: np.ndarray = None):
        """Create (seq_length, n_features) sequences."""
        X = []
        y = []
        for i in range(self.seq_length, len(features)):
            X.append(features[i - self.seq_length : i])
            if targets is not None:
                y.append(targets[i])
        X = np.array(X)
        if targets is not None:
            y = np.array(y)
            return X, y
        return X

    def fit(self, data: pd.DataFrame) -> None:
        if not _HAS_TORCH:
            return

        df = self._prepare_features(data)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
        df = df.dropna(subset=self._feature_cols + ["target"])

        if len(df) < self.seq_length + 100:
            logger.warning(f"LSTM: insufficient data ({len(df)} rows)")
            return

        # Normalise features
        features = df[self._feature_cols].values
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        targets = df["target"].values
        X, y = self._make_sequences(features, targets)

        # Train/val split
        split = int(len(X) * 0.8)
        X_train = torch.FloatTensor(X[:split]).to(self._device)
        y_train = torch.FloatTensor(y[:split]).unsqueeze(1).to(self._device)

        self._model = _LSTMNet(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
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
        self._norm_mean = mean
        self._norm_std = std

        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), MODEL_DIR / "lstm_latest.pt")
        logger.info(f"LSTM trained on {split} sequences, last_loss={total_loss / max(n_batches, 1):.4f}")

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
