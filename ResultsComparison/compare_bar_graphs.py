#!/usr/bin/env python3
"""
Compare experiment CSV files with bar graphs.

Usage examples:
    python3 compare_bar_graphs.py /path/to/all_stats_150.csv /path/to/all_stats_200.csv /path/to/all_stats_300.csv

    python3 compare_bar_graphs.py data/a.csv data/b.csv data/c.csv --labels 150m 200m 300m

    python3 compare_bar_graphs.py results/*.csv --output-dir comparison_plots

    python3 compare_bar_graphs.py a.csv b.csv --metrics PDR "Avg Delay" "#Retransmissions"

What this script does:
- Loads one or more experiment CSV files
- Keeps the per-flow rows (for example rows whose first column contains "Flow")
- Converts numeric columns to floats where possible
- Computes the mean value of each metric for each file
- Creates one bar graph per metric
- Saves all graphs to the chosen output directory

Notes:
- If you do not pass --labels, labels are taken from the CSV filenames.
- By default, all numeric metrics are plotted except "Run No".
- Use --metrics to limit plotting to specific columns.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare experiment CSV files with side-by-side bar graphs."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="CSV files to compare"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Custom labels for the files, in the same order as the file list"
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_plots",
        help="Directory where plots will be saved (default: comparison_plots)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional list of metric column names to plot"
    )
    parser.add_argument(
        "--include-run-no",
        action="store_true",
        help='Include the "Run No" column if present'
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them"
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "metric"


def load_experiment_csv(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    first_col = df.columns[0]
    first_col_as_str = df[first_col].astype(str)

    # Prefer per-flow rows when they exist.
    flow_mask = first_col_as_str.str.contains("Flow", case=False, na=False)
    if flow_mask.any():
        df = df[flow_mask]
    else:
        # Otherwise drop obvious summary/stat rows if present.
        bad_prefixes = (
            "Mean", "Median", "Std", "CI", "Min", "Max",
            "Confidence", "Variance"
        )
        keep_mask = ~first_col_as_str.str.startswith(bad_prefixes, na=False)
        df = df[keep_mask]

    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Keep only columns that actually contain numeric data.
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError(f"No numeric metric columns found in: {path}")

    # Mean over all kept rows.
    return numeric_df.mean(numeric_only=True)


def build_summary(file_paths, labels):
    rows = []
    for label, file_path in zip(labels, file_paths):
        metrics = load_experiment_csv(file_path)
        metrics["Experiment"] = label
        rows.append(metrics)

    summary = pd.DataFrame(rows).set_index("Experiment")
    return summary


def choose_metrics(summary: pd.DataFrame, requested_metrics, include_run_no: bool):
    available = list(summary.columns)

    if requested_metrics:
        missing = [m for m in requested_metrics if m not in available]
        if missing:
            available_str = ", ".join(available)
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Requested metric(s) not found: {missing_str}\n"
                f"Available metrics are: {available_str}"
            )
        metrics = requested_metrics
    else:
        metrics = available

    if not include_run_no:
        metrics = [m for m in metrics if m != "Run No"]

    return metrics


def save_summary_csv(summary: pd.DataFrame, output_dir: Path):
    out_csv = output_dir / "summary_comparison.csv"
    summary.to_csv(out_csv)
    return out_csv


def plot_metric(summary: pd.DataFrame, metric: str, output_dir: Path, show: bool):
    values = summary[metric]

    plt.figure(figsize=(10, 6))
    ax = values.plot(kind="bar")

    plt.title(metric)
    plt.xlabel("Experiment")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Add value labels above bars.
    y_min = float(values.min()) if len(values) else 0.0
    y_max = float(values.max()) if len(values) else 0.0
    y_range = y_max - y_min
    offset = 0.01 * y_range if y_range > 0 else (0.01 * abs(y_max) if y_max != 0 else 0.1)

    for patch, val in zip(ax.patches, values):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        label = f"{val:.6g}" if pd.notna(val) else "NaN"
        va = "bottom" if y >= 0 else "top"
        y_text = y + offset if y >= 0 else y - offset
        ax.text(x, y_text, label, ha="center", va=va, fontsize=9)

    filename = sanitize_filename(metric) + ".png"
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path


def main():
    args = parse_args()

    file_paths = [Path(f).expanduser().resolve() for f in args.files]

    if args.labels:
        if len(args.labels) != len(file_paths):
            raise ValueError(
                f"You passed {len(args.labels)} labels for {len(file_paths)} files. "
                "The counts must match."
            )
        labels = args.labels
    else:
        labels = [p.stem for p in file_paths]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(file_paths, labels)
    metrics_to_plot = choose_metrics(summary, args.metrics, args.include_run_no)

    if not metrics_to_plot:
        raise ValueError("No metrics left to plot after filtering.")

    summary_csv = save_summary_csv(summary, output_dir)

    saved_plots = []
    for metric in metrics_to_plot:
        saved_plots.append(plot_metric(summary, metric, output_dir, args.show))

    print("Comparison complete.")
    print(f"Summary CSV: {summary_csv}")
    print("Saved plot files:")
    for path in saved_plots:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
