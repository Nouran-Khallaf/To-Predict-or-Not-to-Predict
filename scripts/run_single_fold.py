#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run one fold only."""

import argparse

from uncertainty_benchmark.config import load_config
from uncertainty_benchmark.runner import run_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check configured files before running.",
    )
    args = parser.parse_args()

    config = load_config(args.config, check_files=args.check_files)
    run_fold(config, args.fold)


if __name__ == "__main__":
    main()
