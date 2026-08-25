"""Run the dissertation pipeline from data collection through diagnostics.

Run from the repository root with the project virtual environment active:
    python run_pipeline.py

The collection and graph stages are resumable where their underlying scripts
support caching. The runner stops immediately if any stage fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
FEATURES = ROOT / "data" / "config3_features.csv"


CORE_STEPS = [
    ("News coverage probe", "newscoverage.py"),
    ("News collection", "collectnews.py"),
    ("Price features", "build_price.py"),
    ("Sentiment features", "build_sentiment.py"),
    ("Graph features", "build_graph.py"),
    ("Ablation modelling", "ablation.py", str(FEATURES)),
]

ANALYSIS_STEPS = [
    ("DeLong and McNemar tests", "delong.py", str(FEATURES)),
    ("Pipeline audit", "audit.py", str(FEATURES)),
    ("Descriptive analysis", "descrip_cols.py"),
    ("Sector check", "check_sectors.py"),
    ("Graph visualisation", "graph_viz.py"),
    ("Feature diagnostics", "diag_feature.py"),
    ("Diagnostic boxplots", "diag_boxplots.py"),
    ("Diagnostic ROC curves", "diag_roccurves.py"),
    ("Per-stock diagnostics", "diag_perstock.py"),
    ("McNemar diagnostics", "diag_mcnemar.py"),
]


def run_step(label: str, script: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="skip the optional Alpha Vantage coverage probe",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="skip news collection and use existing raw news files",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="run data preparation and modelling, but skip diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_collection and not os.environ.get("ALPHAVANTAGE_API_KEY"):
        raise SystemExit(
            "ALPHAVANTAGE_API_KEY is required for collection. "
            "Set it in PowerShell or use --skip-collection."
        )

    steps = list(CORE_STEPS)
    if args.skip_coverage:
        steps = [step for step in steps if step[1] != "newscoverage.py"]
    if args.skip_collection:
        steps = [
            step for step in steps
            if step[1] not in {"newscoverage.py", "collectnews.py"}
        ]
    if not args.skip_analysis:
        steps.extend(ANALYSIS_STEPS)

    print(f"Project root: {ROOT}")
    print(f"Python interpreter: {sys.executable}")
    print(f"Stages to run: {len(steps)}")

    for label, script, *script_args in steps:
        run_step(label, script, *script_args)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
