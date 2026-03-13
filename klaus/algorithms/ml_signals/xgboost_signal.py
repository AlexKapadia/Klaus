"""XGBoost direction classifier signal."""

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
class XGBoostSignal(BaseAlgorithm):
    """XGBoost classifier predicting next-bar direction.

    Features: returns, rolling vol, RSI, MACD, Bollinger %B, ATR ratio, z-score.
    Target: 1 if next close > current close, else 0.
    Retrains periodically on rolling window.
    """

    name = "xgboost_signal"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_bars_required = 300

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.n_estimators = self.params.get("n_estimators", 200)
        self.max_depth = self.params.get("max_depth", 6)
        self.learning_rate = self.params.get("learning_rate", 0.05)
        self.retrain_days = self.params.get("retrain_interval_days", 7)

        self._model: Optional[xgb.XGBClassifier] = None
        self._last_train_time: Optional[datetime] = None
        self._feature_cols = [
            "returns", "rolling_volatility", "rsi", "macd_hist",
            "bb_pct", "zscore", "atr_ratio",
        ]

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV + indicators."""
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
            df["rsi"] = 100 - (100 / (1 + rs))
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
        if "atr" in df.columns:
            df["atr_ratio"] = df["atr"] / df["close"]
        else:
            high_low = df["high"] - df["low"]
            df["atr_ratio"] = high_low.rolling(14).mean() / df["close"]

        return df

    def fit(self, data: pd.DataFrame) -> None:
        """Train XGBoost on historical data."""
        if not _HAS_XGB:
            logger.warning("xgboost not installed")
            return

        df = self._prepare_features(data)

        # Target: next bar direction
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna(subset=self._feature_cols + ["target"])

        if len(df) < 100:
            logger.warning(f"XGBoost: insufficient data ({len(df)} rows)")
            return

        X = df[self._feature_cols].values
        y = df["target"].values

        # Train on all but last 20% (validation)
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
        )
        self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_acc = self._model.score(X_val, y_val)
        self._last_train_time = datetime.utcnow()
        logger.info(f"XGBoost trained: val_accuracy={val_acc:.3f} on {len(X_train)} samples")

        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(MODEL_DIR / "xgboost_latest.json"))

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not _HAS_XGB or self._model is None:
            return None

        # Check if retrain needed
        if self._needs_retrain():
            self.fit(data)

        df = self._prepare_features(data)
        df = df.dropna(subset=self._feature_cols)

        if len(df) == 0:
            return None

        X = df[self._feature_cols].iloc[[-1]].values
        prob = self._model.predict_proba(X)[0]

        # prob[1] = probability of up move
        prob_up = prob[1]

        if prob_up > 0.6:
            direction = Direction.LONG
            strength = (prob_up - 0.5) * 2  # map 0.5-1.0 to 0-1
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

    def _needs_retrain(self) -> bool:
        if self._last_train_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_train_time).days
        return elapsed >= self.retrain_days
