#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create paper-ready tables from benchmark outputs."""

import argparse

from uncertainty_benchmark.io.tables import create_paper_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to a completed experiment results directory.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for generated tables. Defaults to results-dir/paper_tables.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places.",
    )
    args = parser.parse_args()

    outputs = create_paper_tables(
        results_dir=args.results_dir,
        output_dir=args.outdir,
        decimals=args.decimals,
    )

    print("Saved paper tables:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
