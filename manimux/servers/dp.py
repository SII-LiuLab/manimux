"""Launch the XPolicyLab DP server using the paired experiment recipe."""

import argparse
import sys
from pathlib import Path


def main():
    from manimux.cli import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--local")
    args = parser.parse_args()
    config = load_config(args.experiment, local=args.local)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "XPolicyLab"))
    from XPolicyLab.setup_policy_server import main as serve

    serve(config["policy_server"])


if __name__ == "__main__":
    main()
