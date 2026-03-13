"""DQN reinforcement learning scalper for M1 forex HFT.

Deep Q-Network with epsilon-greedy exploration for high-frequency
forex scalping. Uses a two-branch architecture (LSTM + DNN) adapted
from the commodity DQN scalper for the forex microstructure:
- Branch 1: LSTM on technical features (returns, RSI, ATR)
- Branch 2: DNN on microstructure features (VWAP-z, OFI proxy)

Actions: BUY (1), SELL (2), HOLD (0).
Risk-sensitive reward: position * return / volatility.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Regime, Signal
from stepsister.config.settings import STEPSISTER_ROOT
from stepsister.core.registry import register_fx_algorithm

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

MODEL_DIR = STEPSISTER_ROOT / "data" / "models"


class _FXDQNNetwork(nn.Module):
    """Two-branch DQN: LSTM branch + DNN branch -> fused Q-values."""

    def __init__(self, lstm_input_dim: int, dnn_input_dim: int,
                 lstm_hidden: int = 32, dnn_hidden: int = 64,
                 seq_len: int = 20, n_actions: int = 3):
        super().__init__()
        self.seq_len = seq_len

        # Branch 1: LSTM for sequential features
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True)
        self.lstm_fc = nn.Linear(lstm_hidden, 32)

        # Branch 2: DNN for snapshot features
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, dnn_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dnn_hidden, 32),
            nn.ReLU(),
        )

        # Fusion -> Q-values
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_actions),
        )

    def forward(self, lstm_x, dnn_x):
        # LSTM branch
        lstm_out, _ = self.lstm(lstm_x)
        lstm_feat = torch.relu(self.lstm_fc(lstm_out[:, -1, :]))

        # DNN branch
        dnn_feat = self.dnn(dnn_x)

        # Fuse
        fused = torch.cat([lstm_feat, dnn_feat], dim=1)
        q_values = self.fusion(fused)
        return q_values


class _FXReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list = []
        self.pos = 0

    def push(self, state_lstm, state_dnn, action, reward, next_lstm, next_dnn, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state_lstm, state_dnn, action, reward, next_lstm, next_dnn, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


@register_fx_algorithm
class FXDQNScalper(BaseAlgorithm):
    """Deep Q-Network scalper for HFT forex trading.

    Actions: 0=HOLD, 1=BUY, 2=SELL
    State: (LSTM sequence of technical features, DNN snapshot of micro features)
    Reward: risk-adjusted return = position * bar_return / rolling_vol
    """

    name = "fx_dqn_scalper"
    supported_instruments = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING, Regime.VOLATILE]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.seq_len = self.params.get("sequence_length", 20)
        self.lstm_features = ["returns", "rsi_norm", "atr_norm"]
        self.dnn_features = ["returns", "rolling_vol", "vwap_z", "ofi_proxy", "volume_ratio"]
        self.gamma = self.params.get("gamma", 0.95)
        self.epsilon = self.params.get("epsilon", 0.1)
        self.lr = self.params.get("learning_rate", 0.001)
        self.batch_size = self.params.get("batch_size", 64)
        self.retrain_days = self.params.get("retrain_interval_days", 1)
        self.min_confidence = self.params.get("min_confidence", 0.6)

        self._model: Optional[_FXDQNNetwork] = None
        self._target_model: Optional[_FXDQNNetwork] = None
        self._optimizer = None
        self._buffer = _FXReplayBuffer(20000) if _HAS_TORCH else None
        self._last_train_time: Optional[datetime] = None
        self._trained = False

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()

        # Normalised RSI (7-period for HFT)
        if "rsi" in df.columns:
            df["rsi_norm"] = (df["rsi"] - 50) / 50
        else:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi_norm"] = ((100 - (100 / (1 + rs))) - 50) / 50

        # Normalised ATR
        if "atr" in df.columns:
            atr_std = df["atr"].rolling(20).std().replace(0, np.nan)
            df["atr_norm"] = df["atr"] / df["close"]
        else:
            high_low = df["high"] - df["low"]
            atr = high_low.rolling(10).mean()
            df["atr_norm"] = atr / df["close"]

        # Rolling volatility
        df["rolling_vol"] = df["returns"].rolling(10).std()

        # VWAP z-score (if not pre-computed)
        if "vwap_z" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
            cum_tp_vol = (tp * vol).rolling(30).sum()
            cum_vol = vol.rolling(30).sum().replace(0, np.nan)
            micro_vwap = cum_tp_vol / cum_vol
            deviation = df["close"] - micro_vwap
            dev_std = deviation.rolling(30).std().replace(0, np.nan)
            df["vwap_z"] = deviation / dev_std

        # Order flow imbalance proxy
        if "ofi_proxy" not in df.columns:
            body = df["close"] - df["open"]
            full_range = (df["high"] - df["low"]).replace(0, np.nan)
            df["ofi_proxy"] = body / full_range

        # Volume ratio
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["volume_ratio"] = df["volume"] / avg_vol
        else:
            df["volume_ratio"] = 1.0

        return df

    def fit(self, data: pd.DataFrame) -> None:
        if not _HAS_TORCH:
            return

        df = self._prepare_features(data)
        df = df.dropna()

        if len(df) < self.seq_len + 100:
            return

        lstm_dim = len(self.lstm_features)
        dnn_dim = len(self.dnn_features)

        self._model = _FXDQNNetwork(lstm_dim, dnn_dim, seq_len=self.seq_len)
        self._target_model = _FXDQNNetwork(lstm_dim, dnn_dim, seq_len=self.seq_len)
        self._target_model.load_state_dict(self._model.state_dict())
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

        # Generate experience from historical data
        lstm_cols = [c for c in self.lstm_features if c in df.columns]
        dnn_cols = [c for c in self.dnn_features if c in df.columns]

        lstm_data = df[lstm_cols].values
        dnn_data = df[dnn_cols].values
        returns = df["returns"].values
        vol = df["rolling_vol"].values

        # Fill NaN with 0 for training
        lstm_data = np.nan_to_num(lstm_data, 0)
        dnn_data = np.nan_to_num(dnn_data, 0)
        returns = np.nan_to_num(returns, 0)
        vol = np.nan_to_num(vol, 1e-6)
        vol = np.where(vol <= 0, 1e-6, vol)

        # Populate replay buffer with simulated experience
        for i in range(self.seq_len, len(df) - 1):
            lstm_seq = lstm_data[i - self.seq_len:i]
            dnn_snap = dnn_data[i]
            next_lstm = lstm_data[i - self.seq_len + 1:i + 1]
            next_dnn = dnn_data[i + 1] if i + 1 < len(dnn_data) else dnn_data[i]

            # Simulate: best action is direction of next return
            next_ret = returns[i + 1] if i + 1 < len(returns) else 0
            best_action = 1 if next_ret > 0 else (2 if next_ret < 0 else 0)
            reward = abs(next_ret) / vol[i]

            self._buffer.push(lstm_seq, dnn_snap, best_action, reward,
                              next_lstm, next_dnn, False)

        # Train for several episodes
        if len(self._buffer) < self.batch_size:
            return

        self._model.train()
        loss_fn = nn.MSELoss()

        for epoch in range(min(50, len(self._buffer) // self.batch_size)):
            s_lstm, s_dnn, actions, rewards, ns_lstm, ns_dnn, dones = \
                self._buffer.sample(self.batch_size)

            s_lstm_t = torch.FloatTensor(np.array(s_lstm))
            s_dnn_t = torch.FloatTensor(np.array(s_dnn))
            ns_lstm_t = torch.FloatTensor(np.array(ns_lstm))
            ns_dnn_t = torch.FloatTensor(np.array(ns_dnn))
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            dones_t = torch.BoolTensor(dones)

            # Current Q-values
            q_vals = self._model(s_lstm_t, s_dnn_t)
            q_action = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze()

            # Target Q-values
            with torch.no_grad():
                next_q = self._target_model(ns_lstm_t, ns_dnn_t)
                max_next_q = next_q.max(1)[0]
                target = rewards_t + self.gamma * max_next_q * (~dones_t)

            loss = loss_fn(q_action, target)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # Update target network
        self._target_model.load_state_dict(self._model.state_dict())

        self._last_train_time = datetime.utcnow()
        self._trained = True
        logger.info(f"FX DQN Scalper trained on {len(self._buffer)} experiences")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(MODEL_DIR / "fx_dqn_scalper_latest.pth"))

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not _HAS_TORCH or not self._trained or self._model is None:
            return None

        if self._needs_retrain():
            self.fit(data)

        df = self._prepare_features(data)
        df = df.dropna()

        if len(df) < self.seq_len + 1:
            return None

        lstm_cols = [c for c in self.lstm_features if c in df.columns]
        dnn_cols = [c for c in self.dnn_features if c in df.columns]

        lstm_seq = np.nan_to_num(df[lstm_cols].iloc[-self.seq_len:].values, 0)
        dnn_snap = np.nan_to_num(df[dnn_cols].iloc[-1].values, 0)

        self._model.eval()
        with torch.no_grad():
            lstm_t = torch.FloatTensor(lstm_seq).unsqueeze(0)
            dnn_t = torch.FloatTensor(dnn_snap).unsqueeze(0)
            q_values = self._model(lstm_t, dnn_t)[0]

        # Softmax for confidence
        probs = torch.softmax(q_values, dim=0).numpy()
        action = int(q_values.argmax())

        # Epsilon-greedy during live trading (reduced)
        if np.random.random() < self.epsilon * 0.1:
            return None

        confidence = float(probs[action])
        if confidence < self.min_confidence:
            return None

        # action 0=HOLD, 1=BUY, 2=SELL
        if action == 0:
            return None
        elif action == 1:
            direction = Direction.LONG
        else:
            direction = Direction.SHORT

        # Signal strength from Q-value confidence (max Q - min Q)
        q_range = float(q_values.max() - q_values.min())
        strength = np.clip((confidence - 0.5) * 2, 0.01, 1.0)

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "q_values": q_values.numpy().tolist(),
                "q_range": q_range,
                "confidence": float(confidence),
                "action": action,
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )

    def _needs_retrain(self) -> bool:
        if self._last_train_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_train_time).days
        return elapsed >= self.retrain_days
