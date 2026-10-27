"""Calculate tables from the main experiment's saved records. No model calls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_society.evaluation.analysis import write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/checkpoints/main")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables/main-analysis")
    args = parser.parse_args()
    try:
        result = write_analysis(args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot analyse results: {error}\n")
    print(json.dumps(result["counts"], indent=2))
    print(f"Tables: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
