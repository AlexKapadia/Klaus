@echo off
title Step Sister - AI Forex Trading Platform
color 0A
cd /d "%~dp0"
echo.
echo ============================================================
echo   STEP SISTER - AI-Driven Forex Trading Platform
echo   (Klaus's Step Sister)
echo ============================================================
echo.
echo   [1] Standard  - H1 bars, 60s cycles
echo   [2] HFT       - M1 bars, 1s cycles (max trades)
echo   [3] Both      - Standard + HFT in parallel
echo.
set /p MODE="  Select mode [1/2/3]: "

if "%MODE%"=="1" (
    echo.
    echo  Starting STANDARD engine...
    echo.
    python -m stepsister.main
) else if "%MODE%"=="2" (
    echo.
    echo  Starting HFT engine...
    echo.
    python -m stepsister.main --hft
) else if "%MODE%"=="3" (
    echo.
    echo  Starting DUAL engine (Standard + HFT)...
    echo.
    python -m stepsister.main --both
) else (
    echo  Invalid choice. Defaulting to Standard.
    echo.
    python -m stepsister.main
)

echo.
echo  Step Sister has stopped.
pause
