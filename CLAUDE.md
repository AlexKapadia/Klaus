# Klaus — AI-Driven Algorithmic Trading Platform

## MANDATORY RULES

### External Terminal Launch Rule
**Every time Klaus needs to be started or restarted, ALWAYS launch it in an external terminal using PowerShell:**
```
powershell.exe -Command "Start-Process cmd.exe -ArgumentList '/k cd /d \"C:\Users\alexa\OneDrive - NOCU\Klaus\" && python -m klaus.main --hft'"
```
- NEVER run Klaus inside the Claude Code sandbox terminal
- NEVER ask the user how to launch it — just do it with the command above
- ALWAYS clear `__pycache__` before launching:
  ```
  find "C:/Users/alexa/OneDrive - NOCU/Klaus/klaus" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  ```
- After code changes, ALWAYS restart by clearing cache + launching external terminal
- Use `dangerouslyDisableSandbox: true` on the PowerShell launch command

### Stepsister Launch Rule
**When the user asks to launch "stepsister" (Klaus's Step Sister — Forex trading platform), ALWAYS launch it in an external terminal using PowerShell:**
```
powershell.exe -Command "Start-Process cmd.exe -ArgumentList '/k cd /d \"C:\Users\alexa\OneDrive - NOCU\Klaus\" && python -m stepsister.main --hft'"
```
- Same rules as Klaus: NEVER run inside sandbox, NEVER ask how to launch — just do it
- ALWAYS clear `__pycache__` before launching:
  ```
  find "C:/Users/alexa/OneDrive - NOCU/Klaus/stepsister" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  ```
- Use `dangerouslyDisableSandbox: true` on the PowerShell launch command

### Stepsister Project Structure
- Entry point: `python -m stepsister.main --hft`
- Location: `stepsister/` (inside Klaus repo root)
- Config: `stepsister/config/` — risk.yaml, algorithms.yaml, instruments.yaml, regimes.yaml
- HFT engine: `stepsister/orchestrator/hft_engine.py`
- Standard engine: `stepsister/orchestrator/engine.py`
- FX HFT algos: `stepsister/algorithms/hft/`
- FX ML algos: `stepsister/algorithms/ml/`
- Monitoring/logs: `stepsister/data/logs/`

### Account Details
- Demo account 800000010: TabTrade-Live, ~$1M balance (may have large floating losses reducing equity/margin)
- Demo account 800000011: TabTrade-Live, ~$998 balance (small account, growth mode)

### Project Structure
- Entry point: `python -m klaus.main --hft`
- Config: `klaus/config/` — risk.yaml, algorithms.yaml, instruments.yaml, regimes.yaml
- HFT engine: `klaus/orchestrator/hft_engine.py`
- HFT risk: `klaus/risk/hft_risk_manager.py`
- HFT algos: `klaus/algorithms/hft/` and `klaus/algorithms/ml_hft/`
- MT5 client: `klaus/data/mt5_client.py`
- Market data: `klaus/data/market_data.py`
