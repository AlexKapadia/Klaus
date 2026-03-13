"""Unified launcher — choose Klaus (commodities) or Step Sister (forex).

Usage:
    python run.py klaus              # Klaus standard engine (commodities)
    python run.py klaus --hft        # Klaus HFT engine
    python run.py klaus --both       # Klaus dual mode

    python run.py stepsister         # Step Sister standard engine (forex)
    python run.py stepsister --hft   # Step Sister HFT engine (forex)
    python run.py stepsister --both  # Step Sister dual mode (forex)

    python run.py                    # Interactive menu
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def show_menu():
    """Interactive platform selection."""
    print()
    print("=" * 50)
    print("  TRADING PLATFORM LAUNCHER")
    print("=" * 50)
    print()
    print("  [1] KLAUS         — Commodities (Gold, Oil, Gas, Silver...)")
    print("  [2] STEP SISTER   — Forex (EUR/USD, GBP/USD, USD/JPY...)")
    print()
    print("  Engine modes:")
    print("    [s] Standard  — H1 bars, 60s cycles")
    print("    [h] HFT       — M1 bars, 1s cycles (max trades)")
    print("    [b] Both      — Standard + HFT in parallel")
    print()

    platform = input("  Platform [1/2]: ").strip()
    if platform not in ("1", "2"):
        print("  Invalid choice. Exiting.")
        sys.exit(1)

    mode = input("  Mode [s/h/b]: ").strip().lower()
    mode_map = {"s": "standard", "h": "hft", "b": "both"}
    if mode not in mode_map:
        print("  Invalid mode. Defaulting to HFT.")
        mode = "h"

    engine_mode = mode_map[mode]

    if platform == "1":
        _launch_klaus(engine_mode)
    else:
        _launch_stepsister(engine_mode)


def _launch_klaus(mode: str):
    """Launch Klaus (commodities)."""
    # Patch sys.argv for Klaus's mode parser
    sys.argv = ["klaus.main"]
    if mode == "hft":
        sys.argv.append("--hft")
    elif mode == "both":
        sys.argv.append("--both")

    from klaus.main import main
    main()


def _launch_stepsister(mode: str):
    """Launch Step Sister (forex)."""
    # Patch sys.argv for Step Sister's mode parser
    sys.argv = ["stepsister.main"]
    if mode == "hft":
        sys.argv.append("--hft")
    elif mode == "both":
        sys.argv.append("--both")

    from stepsister.main import main
    main()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        show_menu()
    else:
        platform = args[0].lower()
        # Pass remaining args through
        remaining = args[1:]

        if platform == "klaus":
            mode = "standard"
            if "--hft" in remaining:
                mode = "hft"
            elif "--both" in remaining:
                mode = "both"
            _launch_klaus(mode)

        elif platform in ("stepsister", "step-sister", "sister", "forex"):
            mode = "standard"
            if "--hft" in remaining:
                mode = "hft"
            elif "--both" in remaining:
                mode = "both"
            _launch_stepsister(mode)

        else:
            print(f"Unknown platform: {platform}")
            print("Usage: python run.py [klaus|stepsister] [--hft|--both]")
            sys.exit(1)
