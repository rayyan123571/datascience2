"""
End-to-end pipeline runner.

  python run_pipeline.py              # EDA + training
  python run_pipeline.py --skip-eda   # just (re)train
  python run_pipeline.py --skip-train # just regenerate plots/summary

After it finishes, launch the web app with:
  streamlit run app_streamlit/app.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
APP_DIR = ROOT / "app_streamlit"


def _run(label: str, script: Path) -> None:
    print("\n" + "=" * 70)
    print(f"=> {label}: python {script.relative_to(ROOT)}")
    print("=" * 70)
    rc = subprocess.run(
        [sys.executable, str(script)], cwd=str(script.parent)
    ).returncode
    if rc != 0:
        sys.exit(f"\n{label} failed with exit code {rc}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    eda = APP_DIR / "eda.py"
    train = APP_DIR / "train_model.py"

    if not eda.exists() or not train.exists():
        sys.exit(f"Expected scripts not found under {APP_DIR}")

    if not args.skip_eda:
        _run("EDA", eda)
    if not args.skip_train:
        _run("MODEL TRAINING", train)

    print("\n" + "=" * 70)
    print("[OK] Pipeline finished. Launch the web app with:")
    print("    streamlit run app_streamlit/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
