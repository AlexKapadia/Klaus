"""XGBoost classifier optimised for M1-bar HFT on commodities.

Same core approach as xgboost_signal but adapted for minute-bar data:
- Additional microstructure features (volume ratio, range ratio, OFI proxy)
- Faster retraining cycle (every 4 hours instead of weekly)
- Probability-weighted signal with higher confidence threshold
- Designed for the higher noise environment of sub-hourly data

Inspired by the Dynamic Signal System using sliding-window XGBoost
for natural gas (ResearchGate, 2025) which achieved Sharpe 0.85.
"""

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
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

MODEL_DIR = PROJECT_ROOT / "data" / "models"


@register_algorithm
class XGBoostHFT(BaseAlgorithm):
    """XGBoost direction classifier on M1 bars with microstructure features.

    Higher confidence threshold (0.65) than standard XGBoost signal (0.60)
    due to noisier minute-bar data. Retrains every 4 hours.
    """

    name = "xgboost_hft"
    supported_instruments = ["XAUUSD", "XTIUSD", "XBRUSD", "XNGUSD", "XAGUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_bars_required = 200

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.n_estimators = self.params.get("n_estimators", 150)
        self.max_depth = self.params.get("max_depth", 5)
        self.learning_rate = self.params.get("learning_rate", 0.08)
        self.retrain_hours = self.params.get("retrain_interval_hours", 4)
        self.prob_threshold = self.params.get("probability_threshold", 0.65)
        self.prediction_horizon = self.params.get("prediction_horizon", 5)

        self._model: Optional[xgb.XGBClassifier] = None
        self._last_train_time: Optional[datetime] = None
        self._feature_cols = [
            "returns_1", "returns_3", "returns_5",
            "rolling_vol_5", "rolling_vol_10",
            "rsi_7", "macd_hist_norm",
            "bb_pct", "volume_ratio",
            "range_ratio", "ofi_proxy",
            "price_acceleration", "high_low_ratio",
        ]

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Multi-horizon returns
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_3"] = df["close"].pct_change(3)
        df["returns_5"] = df["close"].pct_change(5)

        # Multi-scale volatility
        r1 = df["close"].pct_change()
        df["rolling_vol_5"] = r1.rolling(5).std()
        df["rolling_vol_10"] = r1.rolling(10).std()

        # Fast RSI (7-period for HFT)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_7"] = 100 - (100 / (1 + rs))

        # Normalised MACD histogram
        ema_fast = df["close"].ewm(span=8).mean()
        ema_slow = df["close"].ewm(span=21).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=5).mean()
        macd_hist = macd - macd_signal
        macd_std = macd_hist.rolling(20).std().replace(0, np.nan)
        df["macd_hist_norm"] = macd_hist / macd_std

        # Bollinger %B
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        bb_range = (4 * std).replace(0, np.nan)
        df["bb_pct"] = (df["close"] - (sma - 2 * std)) / bb_range

        # Volume ratio (current / 20-bar average)
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["volume_ratio"] = df["volume"] / avg_vol
        else:
            df["volume_ratio"] = 1.0

        # Range ratio (current bar range / 10-bar average range)
        bar_range = df["high"] - df["low"]
        avg_range = bar_range.rolling(10).mean().replace(0, np.nan)
        df["range_ratio"] = bar_range / avg_range

        # Order flow imbalance proxy (bar body / full range)
        body = df["close"] - df["open"]
        full_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["ofi_proxy"] = body / full_range

        # Price acceleration (second derivative of price)
        df["price_acceleration"] = r1.diff()

        # High-low ratio (directional bias within bar)
        df["high_low_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

        return df

    def fit(self, data: pd.DataFrame) -> None:
        if not _HAS_XGB:
            logger.warning("xgboost not installed")
            return

        df = self._prepare_features(data)

        # Target: direction over next N bars
        future_return = df["close"].shift(-self.prediction_horizon) / df["close"] - 1
        df["target"] = (future_return > 0).astype(int)

        available_cols = [c for c in self._feature_cols if c in df.columns]
        df = df.dropna(subset=available_cols + ["target"])

        if len(df) < 100:
            logger.warning(f"XGBoost HFT: insufficient data ({len(df)} rows)")
            return

        X = df[available_cols].values
        y = df["target"].values

        # Time-series split: train on first 80%
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_acc = self._model.score(X_val, y_val)
        self._last_train_time = datetime.utcnow()
        logger.info(f"XGBoost HFT trained: val_accuracy={val_acc:.3f} on {len(X_train)} M1 bars")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(MODEL_DIR / "xgboost_hft_latest.json"))

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not _HAS_XGB or self._model is None:
            return None

        if self._needs_retrain():
            self.fit(data)

        df = self._prepare_features(data)
        available_cols = [c for c in self._feature_cols if c in df.columns]
        df = df.dropna(subset=available_cols)

        if len(df) == 0:
            return None

        X = df[available_cols].iloc[[-1]].values
        prob = self._model.predict_proba(X)[0]
        prob_up = prob[1]

        if prob_up > self.prob_threshold:
            direction = Direction.LONG
            strength = (prob_up - 0.5) * 2
        elif prob_up < (1 - self.prob_threshold):
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
            metadata={
                "prob_up": float(prob_up),
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )

    def _needs_retrain(self) -> bool:
        if self._last_train_time is None:
            return True
        elapsed_hours = (datetime.utcnow() - self._last_train_time).total_seconds() / 3600
        return elapsed_hours >= self.retrain_hours
