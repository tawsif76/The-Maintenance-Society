"""Plan, run, or inspect the fixed 80-episode local main experiment."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_society.agents.ollama_client import OllamaError
from maintenance_society.experiments.main import run_main, export_status
from maintenance_society.experiments.schedule import make_schedule, validate_design
from maintenance_society.experiments.storage import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Show the schedule without inference (default)")
    mode.add_argument("--run", action="store_true", help="Run or resume the main experiment")
    mode.add_argument("--status", action="store_true", help="Read saved progress without inference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/checkpoints/main")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    try:
        if args.status:
            report = export_status(args.output_dir, write=False)
            print(f"Complete: {report['complete']}/80; incomplete: {report['incomplete']}; "
                  f"unfinished: {report['unfinished']}; pending: {report['pending']}")
            print(f"Reports: {args.output_dir.resolve()}")
            return 0
        if not args.run:
            experiment = read_json(ROOT / "configs/experiment.json")
            models = read_json(ROOT / "configs/models.json")
            validate_design(experiment, models)
            schedule = make_schedule(args.seed)
            print("Main experiment: 2 models x 4 conditions x 10 episodes = 80 episodes.")
            print("Each episode: 6 agents, 30 rounds, followed by independent peer recognition.")
            for item in models["models"]:
                print(f"Model: {item['tag']}; A=10, B=10, C=10, D=10")
            print("Matched seeds and display orders are reused across models and conditions.")
            print(f"First episode: {schedule[0].episode_id}; seed={schedule[0].seed}")
            print("Maximum requests before corrections: 22,080. Shutdowns reduce this count.")
            print("Plan only. Model requests: 0. Use --run to execute/resume.")
            return 0
        report = run_main(ROOT, args.output_dir, args.seed)
        print(f"Reports: {args.output_dir.resolve()}")
        return 0 if report["pending"] == 0 and report["unfinished"] == 0 else 2
    except KeyboardInterrupt:
        print("Stopped. Saved responses and rounds are preserved. Repeat --run to resume.")
        return 130
    except (OllamaError, ValueError, RuntimeError, OSError) as error:
        print(f"Cannot proceed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
