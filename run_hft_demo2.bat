@echo off
cd /d "%~dp0"
echo Starting Klaus HFT Engine (Account 800000011)...
echo.
set MT5_LOGIN=800000011
set MT5_PASSWORD=Party123$
python -m klaus.main --hft
pause
