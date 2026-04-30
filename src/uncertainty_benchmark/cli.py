"""Command-line interface."""

import argparse
from uncertainty_benchmark.config import load_config
from uncertainty_benchmark.runner import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
