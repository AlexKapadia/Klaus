"""GaussianHMM regime detection with semantic state labeling."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

from klaus.core.types import Regime

# Try hmmlearn first; fall back to manual implementation
try:
    from hmmlearn.hmm import GaussianHMM as _HMMLearn
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False
    logger.warning("hmmlearn not available — using scipy-based fallback HMM")


class HMMRegimeDetector:
    """3-state Hidden Markov Model for market regime detection.

    Features: [log_returns, rolling_volatility]
    States are semantically labeled as TRENDING, MEAN_REVERTING, VOLATILE
    based on fitted means and variances.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self._model = None
        self._state_labels: dict[int, Regime] = {}
        self._last_fit_time: Optional[datetime] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract [log_returns, rolling_volatility] feature matrix."""
        features = pd.DataFrame(index=df.index)

        if "log_returns" in df.columns:
            features["returns"] = df["log_returns"]
        elif "returns" in df.columns:
            features["returns"] = df["returns"]
        else:
            features["returns"] = np.log(df["close"] / df["close"].shift(1))

        if "rolling_volatility" in df.columns:
            features["volatility"] = df["rolling_volatility"]
        else:
            features["volatility"] = features["returns"].rolling(20).std()

        features = features.dropna()
        return features.values, features.index

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the HMM on historical data."""
        X, idx = self._build_features(df)

        if len(X) < 50:
            logger.warning(f"Insufficient data for HMM fit: {len(X)} rows (need >= 50)")
            return

        if _HAS_HMMLEARN:
            self._model = _HMMLearn(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self._model.fit(X)
        else:
            self._model = _ScipyGaussianHMM(
                n_states=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self._model.fit(X)

        # Label states semantically
        self._label_states(X)
        self._last_fit_time = datetime.utcnow()
        self._fitted = True

        logger.info(f"HMM fitted on {len(X)} observations. Labels: {self._state_labels}")

    def predict(self, df: pd.DataFrame) -> Regime:
        """Predict the current regime from the latest data."""
        if not self._fitted:
            return Regime.UNKNOWN

        X, idx = self._build_features(df)
        if len(X) == 0:
            return Regime.UNKNOWN

        if _HAS_HMMLEARN:
            states = self._model.predict(X)
        else:
            states = self._model.predict(X)

        current_state = states[-1]
        return self._state_labels.get(current_state, Regime.UNKNOWN)

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime for every row (for visualisation)."""
        if not self._fitted:
            return pd.Series(dtype=object)

        X, idx = self._build_features(df)
        if len(X) == 0:
            return pd.Series(dtype=object)

        if _HAS_HMMLEARN:
            states = self._model.predict(X)
        else:
            states = self._model.predict(X)

        labels = [self._state_labels.get(s, Regime.UNKNOWN) for s in states]
        return pd.Series(labels, index=idx, name="regime")

    def _label_states(self, X: np.ndarray) -> None:
        """Assign TRENDING / MEAN_REVERTING / VOLATILE labels to HMM states.

        Logic:
        - VOLATILE: highest variance in returns
        - TRENDING: highest absolute mean return (after excluding volatile)
        - MEAN_REVERTING: remaining state
        """
        if _HAS_HMMLEARN:
            means = self._model.means_
            covars = self._model.covars_
            return_var = np.array([covars[i][0, 0] for i in range(self.n_states)])
            return_mean = means[:, 0]
        else:
            means = self._model.means_
            covars = self._model.covariances_
            return_var = np.array([covars[i][0, 0] for i in range(self.n_states)])
            return_mean = means[:, 0]

        # Volatile = highest return variance
        volatile_idx = int(np.argmax(return_var))

        # Trending = highest |mean return| among remaining
        remaining = [i for i in range(self.n_states) if i != volatile_idx]
        abs_means = [abs(return_mean[i]) for i in remaining]
        trending_idx = remaining[int(np.argmax(abs_means))]

        # Mean-reverting = the leftover
        mean_rev_idx = [i for i in range(self.n_states) if i not in (volatile_idx, trending_idx)][0]

        self._state_labels = {
            trending_idx: Regime.TRENDING,
            mean_rev_idx: Regime.MEAN_REVERTING,
            volatile_idx: Regime.VOLATILE,
        }

    def needs_refit(self, interval_days: int = 7) -> bool:
        """Check if the model should be refit."""
        if self._last_fit_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_fit_time).days
        return elapsed >= interval_days


class _ScipyGaussianHMM:
    """Minimal 3-state Gaussian HMM using EM, implemented with scipy/numpy.

    Fallback when hmmlearn is not installed (e.g., Python 3.13).
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_state)

        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self._transmat: Optional[np.ndarray] = None
        self._startprob: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """Fit via Baum-Welch (EM) algorithm."""
        T, D = X.shape
        K = self.n_states

        # Initialise with K-means-like seeding
        indices = self.rng.choice(T, K, replace=False)
        self.means_ = X[indices].copy()
        self.covariances_ = np.array([np.eye(D) * np.var(X, axis=0) for _ in range(K)])
        self._transmat = np.ones((K, K)) / K
        self._startprob = np.ones(K) / K

        for iteration in range(self.n_iter):
            # E-step: forward-backward
            log_likelihoods = self._compute_log_likelihood(X)
            log_alpha = self._forward(log_likelihoods)
            log_beta = self._backward(log_likelihoods)

            # Posteriors
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_rows(log_gamma)
            gamma = np.exp(log_gamma)

            # M-step
            gamma_sum = gamma.sum(axis=0) + 1e-10

            # Update start probabilities
            self._startprob = gamma[0] / gamma[0].sum()

            # Update transition matrix
            for i in range(K):
                for j in range(K):
                    numerator = 0.0
                    for t in range(T - 1):
                        numerator += np.exp(
                            log_alpha[t, i]
                            + np.log(self._transmat[i, j] + 1e-300)
                            + log_likelihoods[t + 1, j]
                            + log_beta[t + 1, j]
                            - _logsumexp(log_alpha[T - 1])
                        )
                    self._transmat[i, j] = numerator
                row_sum = self._transmat[i].sum()
                if row_sum > 0:
                    self._transmat[i] /= row_sum

            # Update means and covariances
            for k in range(K):
                w = gamma[:, k]
                self.means_[k] = np.average(X, weights=w, axis=0)
                diff = X - self.means_[k]
                self.covariances_[k] = (diff.T @ np.diag(w) @ diff) / gamma_sum[k]
                # Regularise
                self.covariances_[k] += np.eye(D) * 1e-6

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for most likely state sequence."""
        T, D = X.shape
        K = self.n_states

        log_likelihoods = self._compute_log_likelihood(X)

        # Viterbi
        V = np.zeros((T, K))
        path = np.zeros((T, K), dtype=int)

        V[0] = np.log(self._startprob + 1e-300) + log_likelihoods[0]

        for t in range(1, T):
            for j in range(K):
                probs = V[t - 1] + np.log(self._transmat[:, j] + 1e-300)
                path[t, j] = np.argmax(probs)
                V[t, j] = probs[path[t, j]] + log_likelihoods[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(V[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = path[t + 1, states[t + 1]]

        return states

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log p(x_t | state_k) for all t, k."""
        T = X.shape[0]
        K = self.n_states
        ll = np.zeros((T, K))

        for k in range(K):
            try:
                rv = sp_stats.multivariate_normal(
                    mean=self.means_[k],
                    cov=self.covariances_[k],
                    allow_singular=True,
                )
                ll[:, k] = rv.logpdf(X)
            except np.linalg.LinAlgError:
                ll[:, k] = -1e10

        return ll

    def _forward(self, log_lik: np.ndarray) -> np.ndarray:
        T, K = log_lik.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self._startprob + 1e-300) + log_lik[0]

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = _logsumexp(
                    log_alpha[t - 1] + np.log(self._transmat[:, j] + 1e-300)
                ) + log_lik[t, j]

        return log_alpha

    def _backward(self, log_lik: np.ndarray) -> np.ndarray:
        T, K = log_lik.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = _logsumexp(
                    np.log(self._transmat[i] + 1e-300) + log_lik[t + 1] + log_beta[t + 1]
                )

        return log_beta


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    if c == -np.inf:
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


def _logsumexp_rows(x: np.ndarray) -> np.ndarray:
    """Log-sum-exp across columns (for each row)."""
    c = x.max(axis=1, keepdims=True)
    return c + np.log(np.sum(np.exp(x - c), axis=1, keepdims=True))
