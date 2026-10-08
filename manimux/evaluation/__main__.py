"""Build one versioned HTML evidence report per explicitly selected task."""

from __future__ import annotations

import argparse
from pathlib import Path

from manimux.evaluation.report import generate_task_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        nargs="+",
        required=True,
        help="task directories, each containing task.yaml",
    )
    args = parser.parse_args()
    for task_dir in args.task_dir:
        print(generate_task_report(task_dir))


if __name__ == "__main__":
    main()
