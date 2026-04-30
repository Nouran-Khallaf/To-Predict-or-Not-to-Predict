#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise rejection/selective-prediction results at a target coverage."
    )
    parser.add_argument(
        "--curve",
        required=True,
        help="Path to rejection_curve.csv",
    )
    parser.add_argument(
        "--target_coverage",
        type=float,
        default=0.80,
        help="Target coverage to summarise, e.g. 0.80",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the selected summary CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    curve_path = Path(args.curve)
    df = pd.read_csv(curve_path)

    required_cols = [
        "method",
        "requested_coverage",
        "actual_coverage",
        "threshold",
        "accepted_n",
        "rejected_n",
        "total_n",
        "accepted_accuracy",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {curve_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    target = args.target_coverage

    # Baseline = no rejection, usually requested_coverage == 1.0.
    baseline_candidates = df.loc[
        (df["requested_coverage"] - 1.0).abs()
        == (df["requested_coverage"] - 1.0).abs().min()
    ]

    baseline_accuracy = baseline_candidates["accepted_accuracy"].iloc[0]
    total_n = int(baseline_candidates["total_n"].iloc[0])

    # Pick the row closest to the requested target coverage for each method.
    selected_rows = []

    for method, method_df in df.groupby("method"):
        method_df = method_df.copy()
        method_df["coverage_distance"] = (
            method_df["requested_coverage"] - target
        ).abs()

        best_row = method_df.sort_values(
            ["coverage_distance", "actual_coverage"],
            ascending=[True, False],
        ).iloc[0]

        selected_rows.append(best_row)

    selected = pd.DataFrame(selected_rows)

    selected = selected.sort_values(
        "accepted_accuracy",
        ascending=False,
    ).reset_index(drop=True)

    # Clean reporting table.
    report = selected[
        [
            "method",
            "accepted_n",
            "rejected_n",
            "threshold",
            "accepted_accuracy",
            "actual_coverage",
            "rejection_rate",
            "total_n",
        ]
    ].copy()

    report = report.rename(
        columns={
            "method": "Method",
            "accepted_n": "Accepted",
            "rejected_n": "Rejected",
            "threshold": "Threshold",
            "accepted_accuracy": "Accepted accuracy",
            "actual_coverage": "Actual coverage",
            "rejection_rate": "Rejection rate",
            "total_n": "Total",
        }
    )

    best = report.iloc[0]

    print()
    print(f"At around {target:.0%} coverage, the results are:")
    print(
        report[
            ["Method", "Accepted", "Rejected", "Threshold", "Accepted accuracy"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print()
    print(f"So for this selected setting, {best['Method']} is best.")
    print()

    print("Interpretation:")
    print(f"Without rejection: accuracy = {baseline_accuracy:.4f}")
    print(
        f"With {best['Method']} rejection at {target:.0%} coverage: "
        f"accuracy on accepted examples = {best['Accepted accuracy']:.4f}"
    )
    print(
        f"Rejected examples = {int(best['Rejected'])} / {int(best['Total'])} "
        f"≈ {best['Rejection rate']:.0%}"
    )

    print()
    print("Plain meaning:")
    print(
        "The model keeps the examples where it is more confident and rejects "
        "the examples where uncertainty is above the threshold. The accepted "
        "subset is therefore smaller, but more reliable."
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        print()
        print(f"Saved summary table to: {output_path}")


if __name__ == "__main__":
    main()