# Klaus

AI-driven algorithmic trading platform for commodities and forex, built on MetaTrader 5.

Klaus trades commodities (gold, oil, natural gas, silver, platinum, palladium, aluminium). Its companion module **Stepsister** trades major and cross forex pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP, EURJPY, GBPJPY).

Both engines support standard and high-frequency trading modes with real-time execution via MT5.

## Architecture

```
klaus/                  # Commodities trading engine
  algorithms/
    hft/                # Tick scalper, momentum, spread scalper, microstructure reversion,
                        #   order flow imbalance, volatility breakout, cross-commodity lead-lag
    ml_hft/             # DQN scalper, XGBoost HFT
    ml_signals/         # LSTM, TCN, XGBoost signal generators
    mean_reversion/     # Bollinger bands, Z-score
    momentum/           # SMA crossover
    stat_arb/           # Gold-silver ratio, soybean crush spread, spread OU
    volatility/         # GJR-GARCH, HAR
    seasonal/           # Agricultural growing season, natural gas calendar
    geopolitical/       # GPR index
  orchestrator/         # Standard + HFT engines, scheduler, rules engine
  risk/                 # Position sizing, drawdown control, trailing stops,
                        #   correlation filter, HFT risk manager
  data/                 # MT5 client, market data feed, feature stores
  adaptive/             # Algo selector, performance tracker, trade memory
  regime/               # HMM regime detector
  monitoring/           # Logging + metrics

stepsister/             # Forex trading engine
  algorithms/
    hft/                # FX tick scalper, momentum, spread scalper, micro reversion,
                        #   order flow, volatility breakout, cross-pair lead-lag
    ml/                 # FX DQN scalper, LSTM, XGBoost HFT, XGBoost signal
    mean_reversion/     # FX Bollinger, Z-score
    momentum/           # FX momentum, SMA crossover
    carry/              # Carry trade, FEER carry
    correlation/        # Commodity-FX correlation
  orchestrator/         # Standard + HFT engines
```

## Setup

**Requirements:** Python 3.12+, MetaTrader 5 terminal running

```bash
pip install -r requirements.txt
```

Create a `.env` file with your MT5 credentials:

```
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
```

## Usage

**Klaus (commodities):**
```bash
python -m klaus.main --hft
```

**Stepsister (forex):**
```bash
python -m stepsister.main --hft
```

## Key Dependencies

- **MetaTrader5** — broker connectivity and order execution
- **PyTorch** — LSTM and DQN models
- **XGBoost** — gradient-boosted signal generation
- **hmmlearn** — Hidden Markov Model regime detection
- **arch** — GARCH volatility modelling
- **scikit-learn** — ML utilities
- **pandas / numpy / scipy** — data processing and statistics
- **loguru** — structured logging

## Configuration

All configs live in `klaus/config/` and `stepsister/config/`:

- `instruments.yaml` — tradeable instruments and lot sizes
- `algorithms.yaml` — algorithm parameters and enable/disable flags
- `risk.yaml` — position limits, drawdown thresholds, exposure caps
- `regimes.yaml` — market regime detection parameters
