@echo off
cd /d "%~dp0"
echo Starting Klaus HFT Engine...
echo.
python -m klaus.main --hft
pause
