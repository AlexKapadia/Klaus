"""XGBoost classifier optimised for M1-bar HFT on forex pairs.

Same core approach as Klaus's xgboost_hft but adapted for the forex
microstructure: multi-horizon returns, VWAP deviation, order flow
imbalance proxy, and realised volatility at multiple scales.

Predicts direction over a short horizon (5 bars) with a probability
threshold to filter noise. Retrains every 4 hours on rolling M1 data.

Designed for the higher noise environment of sub-minute forex data
where tick volume is unreliable but price microstructure is rich.
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
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

MODEL_DIR = STEPSISTER_ROOT / "data" / "models"


@register_fx_algorithm
class FXXGBoostHFT(BaseAlgorithm):
    """XGBoost direction classifier on M1 forex bars with microstructure features.

    Higher confidence threshold (0.55 default) than standard XGBoost
    due to noisier minute-bar data. Retrains every 4 hours.

    Features: multi-horizon returns, RSI, ATR, VWAP-z, OFI proxy,
    volume ratio, realised volatility (5-bar and 10-bar).
    """

    name = "fx_xgboost_hft"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING, Regime.VOLATILE]
    min_bars_required = 50

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.n_estimators = self.params.get("n_estimators", 150)
        self.max_depth = self.params.get("max_depth", 5)
        self.learning_rate = self.params.get("learning_rate", 0.08)
        self.retrain_hours = self.params.get("retrain_interval_hours", 4)
        self.prob_threshold = self.params.get("probability_threshold", 0.55)
        self.prediction_horizon = self.params.get("prediction_horizon", 5)

        self._model: Optional[xgb.XGBClassifier] = None
        self._last_train_time: Optional[datetime] = None
        self._feature_cols = [
            "returns_1", "returns_3", "returns_5",
            "rsi_7", "atr_ratio",
            "vwap_z", "ofi_proxy",
            "volume_ratio",
            "rv_5", "rv_10",
        ]

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Multi-horizon returns
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_3"] = df["close"].pct_change(3)
        df["returns_5"] = df["close"].pct_change(5)

        # Fast RSI (7-period for HFT)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_7"] = 100 - (100 / (1 + rs))

        # ATR ratio (price-normalised)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_ratio"] = tr.rolling(10).mean() / df["close"]

        # VWAP z-score
        if "vwap_z" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
            cum_tp_vol = (tp * vol).rolling(30).sum()
            cum_vol = vol.rolling(30).sum().replace(0, np.nan)
            micro_vwap = cum_tp_vol / cum_vol
            deviation = df["close"] - micro_vwap
            dev_std = deviation.rolling(30).std().replace(0, np.nan)
            df["vwap_z"] = deviation / dev_std

        # Order flow imbalance proxy (bar body / full range)
        if "ofi_proxy" not in df.columns:
            body = df["close"] - df["open"]
            full_range = (df["high"] - df["low"]).replace(0, np.nan)
            df["ofi_proxy"] = body / full_range

        # Volume ratio (current / 20-bar average)
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["volume_ratio"] = df["volume"] / avg_vol
        else:
            df["volume_ratio"] = 1.0

        # Realised volatility at multiple scales (squared returns)
        r1 = df["close"].pct_change()
        sq_returns = r1 ** 2
        df["rv_5"] = np.sqrt(sq_returns.rolling(5).sum())
        df["rv_10"] = np.sqrt(sq_returns.rolling(10).sum())

        return df

    def fit(self, data: pd.DataFrame) -> None:
        if not _HAS_XGB:
            logger.warning("xgboost not installed — fx_xgboost_hft disabled")
            return

        df = self._prepare_features(data)

        # Target: direction over next prediction_horizon bars
        future_return = df["close"].shift(-self.prediction_horizon) / df["close"] - 1
        df["target"] = (future_return > 0).astype(int)

        available_cols = [c for c in self._feature_cols if c in df.columns]
        df = df.dropna(subset=available_cols + ["target"])

        if len(df) < 100:
            logger.warning(f"FX XGBoost HFT: insufficient data ({len(df)} rows)")
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
        logger.info(f"FX XGBoost HFT trained: val_accuracy={val_acc:.3f} on {len(X_train)} M1 bars")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(MODEL_DIR / "fx_xgboost_hft_latest.json"))

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

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "prob_up": float(prob_up),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )

    def _needs_retrain(self) -> bool:
        if self._last_train_time is None:
            return True
        elapsed_hours = (datetime.utcnow() - self._last_train_time).total_seconds() / 3600
        return elapsed_hours >= self.retrain_hours
