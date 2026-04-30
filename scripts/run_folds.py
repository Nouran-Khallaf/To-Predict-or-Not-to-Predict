#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run all folds for one experiment config."""

import argparse

from uncertainty_benchmark.config import load_config
from uncertainty_benchmark.runner import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check that all configured train/prediction files exist before running.",
    )
    args = parser.parse_args()

    config = load_config(args.config, check_files=args.check_files)
    run_experiment(config)


if __name__ == "__main__":
    main()
