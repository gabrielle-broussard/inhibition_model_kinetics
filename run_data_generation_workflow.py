"""
run_data_generation_workflow.py

Basic use:
    python run_data_generation_workflow.py

Run only the time-series CSV workflow:
    python run_data_generation_workflow.py --skip_seeded_models --run_timeseries

Run seeded synthetic model outputs and time-series CSV outputs:
    python run_data_generation_workflow.py --run_timeseries
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label, cmd, project_dir, continue_on_error=False):
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print(" ".join(str(x) for x in cmd))

    result = subprocess.run(cmd, cwd=project_dir, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        message = f"Step failed: {label} | return code {result.returncode}"
        if continue_on_error:
            print(message)
            print("Continuing because --continue_on_error was set.")
        else:
            raise RuntimeError(message)


def existing_script(project_dir, filename):
    path = Path(project_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    return str(path.name)


def main():
    parser = argparse.ArgumentParser(description="Run the data-only inhibition kinetics workflow.")
    parser.add_argument("--project_dir", default=".", help="Main InhibitionKinetics project folder.")
    parser.add_argument(
        "--model_script",
        default="generate_model_recovery_data.py",
        help="Seeded model data-generation script.",
    )
    parser.add_argument(
        "--timeseries_script",
        default="generate_timeseries_data.py",
        help="Optional all-model time-series CSV-generation script.",
    )
    parser.add_argument(
        "--skip_seeded_models",
        action="store_true",
        help="Skip the seeded synthetic model recovery outputs.",
    )
    parser.add_argument(
        "--run_timeseries",
        action="store_true",
        help="Also generate all-model time-series CSV outputs.",
    )
    parser.add_argument("--continue_on_error", action="store_true")

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Project folder does not exist: {project_dir}")

    steps = []

    if not args.skip_seeded_models:
        steps.append(
            (
                "1. Generate seeded synthetic model CSV outputs",
                [sys.executable, existing_script(project_dir, args.model_script)],
            )
        )

    if args.run_timeseries:
        steps.append(
            (
                "2. Generate all-model time-series CSV outputs",
                [sys.executable, existing_script(project_dir, args.timeseries_script)],
            )
        )

    if not steps:
        print("No steps selected. Nothing to run.")
        return

    print(f"Project folder: {project_dir}")
    print("Selected data-generation steps:")
    for label, _ in steps:
        print(f"  - {label}")

    for label, cmd in steps:
        run_step(label, cmd, project_dir, continue_on_error=args.continue_on_error)

    print("\n" + "=" * 90)
    print("Data-only inhibition kinetics workflow complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
