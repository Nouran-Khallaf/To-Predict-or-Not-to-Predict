#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise best rejection method at several coverage levels."
    )
    parser.add_argument(
        "--curve",
        required=True,
        help="Path to rejection_curve.csv",
    )
    parser.add_argument(
        "--coverages",
        nargs="+",
        type=float,
        default=[0.90, 0.80, 0.70, 0.60, 0.50],
        help="Coverage levels to summarise.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.curve)

    required = [
        "method",
        "requested_coverage",
        "actual_coverage",
        "threshold",
        "accepted_n",
        "rejected_n",
        "total_n",
        "accepted_accuracy",
        "rejection_rate",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

    # Baseline/no-rejection accuracy.
    baseline_df = df.loc[
        (df["requested_coverage"] - 1.0).abs()
        == (df["requested_coverage"] - 1.0).abs().min()
    ]

    baseline_accuracy = float(baseline_df["accepted_accuracy"].iloc[0])
    total_n = int(baseline_df["total_n"].iloc[0])

    rows = []

    for target in args.coverages:
        target_rows = []

        for method, method_df in df.groupby("method"):
            method_df = method_df.copy()
            method_df["distance"] = (
                method_df["requested_coverage"] - target
            ).abs()

            row = method_df.sort_values(
                ["distance", "accepted_accuracy"],
                ascending=[True, False],
            ).iloc[0]

            target_rows.append(row)

        target_df = pd.DataFrame(target_rows)

        # Best method at this coverage.
        best = target_df.sort_values(
            ["accepted_accuracy", "actual_coverage"],
            ascending=[False, False],
        ).iloc[0]

        rows.append(
            {
                "target_coverage": target,
                "best_method": best["method"],
                "accepted_accuracy": best["accepted_accuracy"],
                "baseline_accuracy": baseline_accuracy,
                "accuracy_gain": best["accepted_accuracy"] - baseline_accuracy,
                "accepted_n": int(best["accepted_n"]),
                "rejected_n": int(best["rejected_n"]),
                "total_n": int(best["total_n"]),
                "actual_coverage": best["actual_coverage"],
                "rejection_rate": best["rejection_rate"],
                "threshold": best["threshold"],
            }
        )

    summary = pd.DataFrame(rows)

    print()
    print(f"Baseline/no-rejection accuracy: {baseline_accuracy:.4f}")
    print(f"Total examples: {total_n}")
    print()

    display = summary.copy()
    display["target_coverage"] = display["target_coverage"].map(lambda x: f"{x:.0%}")
    display["actual_coverage"] = display["actual_coverage"].map(lambda x: f"{x:.1%}")
    display["rejection_rate"] = display["rejection_rate"].map(lambda x: f"{x:.1%}")
    display["accepted_accuracy"] = display["accepted_accuracy"].map(lambda x: f"{x:.4f}")
    display["baseline_accuracy"] = display["baseline_accuracy"].map(lambda x: f"{x:.4f}")
    display["accuracy_gain"] = display["accuracy_gain"].map(lambda x: f"{x:+.4f}")
    display["threshold"] = display["threshold"].map(lambda x: f"{x:.6f}")

    print("Best method by coverage:")
    print(
        display[
            [
                "target_coverage",
                "best_method",
                "accepted_accuracy",
                "accuracy_gain",
                "accepted_n",
                "rejected_n",
                "rejection_rate",
                "threshold",
            ]
        ].to_string(index=False)
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()