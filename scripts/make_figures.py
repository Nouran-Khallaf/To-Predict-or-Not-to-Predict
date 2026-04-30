#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create standard figures from benchmark outputs."""

import argparse

from uncertainty_benchmark.visualisation import make_all_figures


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
        help="Output directory for figures. Defaults to results-dir/figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for raster formats such as PNG.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf",
    )
    args = parser.parse_args()

    outputs = make_all_figures(
        results_dir=args.results_dir,
        output_dir=args.outdir,
        dpi=args.dpi,
        formats=args.formats,
    )

    print("Saved figures:")
    for name, paths in outputs.items():
        print(f"  {name}:")
        if isinstance(paths, dict):
            for fmt, path in paths.items():
                print(f"    {fmt}: {path}")
        else:
            print(f"    {paths}")


if __name__ == "__main__":
    main()
