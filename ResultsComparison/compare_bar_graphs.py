#!/usr/bin/env python3
"""
Compare experiment results with bar graphs and optional delay-over-time overlays.

You can provide inputs directly on the command line, auto-discover them, or use a YAML/JSON config file.

Examples

1) Config file
    python3 compare_bar_graphs.py --config comparison_config.yaml

2) Direct summary CSV comparison
    python3 compare_bar_graphs.py stats/a.csv stats/b.csv stats/c.csv

3) Auto-discover summary CSVs under a parent folder
    python3 compare_bar_graphs.py --auto-find-csvs ./Results --label-mode parent

4) Auto-discover summary CSVs and delay folders
    python3 compare_bar_graphs.py --auto-find-csvs ./Results --auto-find-delay-dirs ./Results --label-mode parent
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_config_file(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()

    if suffix == ".json":
        return json.loads(text)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config files require PyYAML. Install it with:\n"
                "    pip install pyyaml\n"
                "Or use a .json config file instead."
            ) from exc

        loaded = yaml.safe_load(text)
        return loaded or {}

    raise ValueError("Config file must end in .yaml, .yml, or .json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare experiment CSV files with side-by-side bar graphs and optional delay overlays."
    )

    parser.add_argument(
        "--config",
        help="YAML or JSON config file containing files, labels, delay_dirs, delay_labels, and options"
    )

    # Manual summary CSV inputs
    parser.add_argument(
        "files",
        nargs="*",
        help="Summary CSV files to compare with bar graphs"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Custom labels for summary CSV files, in the same order as the file list"
    )

    # Output and bar options
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where outputs will be saved"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional list of summary metric column names to plot"
    )
    parser.add_argument(
        "--include-run-no",
        action="store_true",
        default=None,
        help='Include the "Run No" column if present'
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Show plots interactively in addition to saving them"
    )
    parser.add_argument(
        "--skip-bar-plots",
        action="store_true",
        default=None,
        help="Skip summary bar graph generation"
    )

    # Manual delay overlay inputs
    parser.add_argument(
        "--delay-dirs",
        nargs="+",
        help="Experiment folders that contain Flow_Node*.csv files for delay-over-time overlay"
    )
    parser.add_argument(
        "--delay-labels",
        nargs="+",
        help="Custom labels for delay overlay folders, in the same order as --delay-dirs"
    )
    parser.add_argument(
        "--delay-window",
        type=int,
        default=None,
        help="Rolling-average window for delay overlay"
    )
    parser.add_argument(
        "--delay-nth",
        type=int,
        default=None,
        help="Plot every nth raw delay point in the overlay scatter"
    )
    parser.add_argument(
        "--delay-scatter-alpha",
        type=float,
        default=None,
        help="Scatter transparency for raw delay points"
    )
    parser.add_argument(
        "--delay-recursive",
        action="store_true",
        default=None,
        help="Recursively search each delay directory for Flow_Node*.csv files"
    )

    # Auto-discovery
    parser.add_argument(
        "--auto-find-csvs",
        nargs="+",
        help="One or more root directories to recursively search for all_stats.csv and all_stats_*.csv"
    )
    parser.add_argument(
        "--auto-find-delay-dirs",
        nargs="+",
        help="One or more root directories to recursively search for experiment folders containing Flow_Node*.csv"
    )
    parser.add_argument(
        "--label-mode",
        choices=["stem", "parent", "path"],
        default=None,
        help="How to auto-generate labels when labels are not supplied"
    )
    parser.add_argument(
        "--filter",
        dest="name_filter",
        default=None,
        help="Optional substring filter applied to discovered paths"
    )
    parser.add_argument(
        "--sort",
        choices=["name", "mtime"],
        default=None,
        help="Sort discovered items by name or modified time"
    )

    return parser.parse_args()


def config_get(config: dict, key: str, default=None):
    return config[key] if key in config and config[key] is not None else default


def merge_args_with_config(args, config: dict):
    """
    Command-line arguments override config values when explicitly provided.
    Config values override built-in defaults.
    """

    defaults = {
        "files": [],
        "labels": None,
        "output_dir": "comparison_plots",
        "metrics": None,
        "include_run_no": False,
        "show": False,
        "skip_bar_plots": False,
        "delay_dirs": None,
        "delay_labels": None,
        "delay_window": 100,
        "delay_nth": 5,
        "delay_scatter_alpha": 0.2,
        "delay_recursive": False,
        "auto_find_csvs": None,
        "auto_find_delay_dirs": None,
        "label_mode": "stem",
        "name_filter": None,
        "sort": "name",
    }

    merged = argparse.Namespace()

    # Positional files are special because argparse defaults to [].
    merged.files = args.files if args.files else config_get(config, "files", defaults["files"])

    for key, default in defaults.items():
        if key == "files":
            continue

        arg_value = getattr(args, key)
        config_value = config_get(config, key, default)

        # For store_true arguments, argparse cannot tell whether False was explicit.
        # Using default=None above means True only appears when supplied on CLI.
        if arg_value is not None:
            setattr(merged, key, arg_value)
        else:
            setattr(merged, key, config_value)

    return merged


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "metric"


def get_base_packet_id(packet_id: str) -> str:
    match = re.match(r"(.+)_\w+$", str(packet_id))
    return match.group(1) if match else str(packet_id)


def make_label(path: Path, mode: str, root=None) -> str:
    if mode == "stem":
        return path.stem
    if mode == "parent":
        return path.parent.name if path.is_file() else path.name
    if mode == "path" and root is not None:
        try:
            return str(path.relative_to(root))
        except Exception:
            return str(path)
    return path.stem


def sort_paths(paths, sort_mode: str):
    if sort_mode == "mtime":
        return sorted(paths, key=lambda p: p.stat().st_mtime)
    return sorted(paths, key=lambda p: str(p).lower())


def discover_summary_csvs(roots, name_filter=None, sort_mode="name", output_dir=None):
    matches = []
    seen = set()

    for root in roots:
        root = Path(root).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"Auto-find CSV root not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Auto-find CSV root is not a directory: {root}")

        for pattern in ("all_stats.csv", "all_stats_*.csv"):
            for path in root.rglob(pattern):
                resolved = path.resolve()

                if output_dir is not None:
                    try:
                        resolved.relative_to(output_dir.resolve())
                        continue
                    except Exception:
                        pass

                if name_filter and name_filter.lower() not in str(resolved).lower():
                    continue

                if resolved not in seen:
                    seen.add(resolved)
                    matches.append(resolved)

    return sort_paths(matches, sort_mode)


def discover_delay_dirs(roots, name_filter=None, sort_mode="name"):
    matches = []
    seen = set()

    for root in roots:
        root = Path(root).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"Auto-find delay root not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Auto-find delay root is not a directory: {root}")

        for path in root.rglob("Flow_Node*.csv"):
            parent = path.parent.resolve()

            if name_filter and name_filter.lower() not in str(parent).lower():
                continue

            if parent not in seen:
                seen.add(parent)
                matches.append(parent)

    return sort_paths(matches, sort_mode)


def load_experiment_csv(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    first_col = df.columns[0]
    first_col_as_str = df[first_col].astype(str)

    flow_mask = first_col_as_str.str.contains("Flow", case=False, na=False)
    if flow_mask.any():
        df = df[flow_mask]
    else:
        bad_prefixes = (
            "Mean", "Median", "Std", "CI", "Min", "Max",
            "Confidence", "Variance", "95% CI Lower", "95% CI Upper"
        )
        keep_mask = ~first_col_as_str.str.startswith(bad_prefixes, na=False)
        df = df[keep_mask]

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError(f"No numeric metric columns found in: {path}")

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
            raise ValueError(
                f"Requested metric(s) not found: {', '.join(missing)}\n"
                f"Available metrics are: {', '.join(available)}"
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

    out_path = output_dir / f"{sanitize_filename(metric)}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path


def find_flow_csvs(experiment_dir: Path, recursive: bool):
    pattern = "**/Flow_Node*.csv" if recursive else "Flow_Node*.csv"
    return sorted(experiment_dir.glob(pattern))


def load_delay_points_for_experiment(experiment_dir: Path, recursive: bool):
    flow_csvs = find_flow_csvs(experiment_dir, recursive=recursive)

    if not flow_csvs:
        raise ValueError(f"No Flow_Node*.csv files found in {experiment_dir}")

    all_frames = []

    for csv_path in flow_csvs:
        df = pd.read_csv(csv_path)

        required_cols = {"Pkt ID", "Delay (in s)", "Generation Time (in s)"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required column(s) {sorted(missing)} in flow CSV: {csv_path}"
            )

        df = df.copy()
        df["Base Packet ID"] = df["Pkt ID"].apply(get_base_packet_id)
        df = df.groupby("Base Packet ID").tail(1).reset_index(drop=True)

        df["Delay (in s)"] = pd.to_numeric(df["Delay (in s)"], errors="coerce")
        df["Generation Time (in s)"] = pd.to_numeric(df["Generation Time (in s)"], errors="coerce")

        df = df[
            df["Delay (in s)"].notna()
            & df["Generation Time (in s)"].notna()
            & (df["Delay (in s)"] > 0)
        ]

        if df.empty:
            continue

        df = df[df["Delay (in s)"] != float("inf")]
        if df.empty:
            continue

        df["Reception Time (s)"] = df["Generation Time (in s)"] + df["Delay (in s)"]
        all_frames.append(df[["Reception Time (s)", "Delay (in s)"]])

    if not all_frames:
        raise ValueError(f"No valid packet-delay rows found in {experiment_dir}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("Reception Time (s)").reset_index(drop=True)
    return combined


def plot_delay_overlay(
    experiment_dirs,
    labels,
    output_dir: Path,
    nth: int,
    scatter_alpha: float,
    window: int,
    recursive: bool,
    show: bool,
):
    plt.figure(figsize=(10, 6))

    for experiment_dir, label in zip(experiment_dirs, labels):
        df = load_delay_points_for_experiment(experiment_dir, recursive=recursive)

        scatter_sample = df.iloc[::max(1, nth), :]
        plt.scatter(
            scatter_sample["Reception Time (s)"],
            scatter_sample["Delay (in s)"],
            s=8,
            alpha=scatter_alpha,
            label="_nolegend_",
        )

        rolling = df["Delay (in s)"].rolling(window=max(1, window), min_periods=1).mean()
        plt.plot(
            df["Reception Time (s)"],
            rolling,
            linewidth=2.5,
            label=label,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Delay (s)")
    plt.title("Packet Delay over Time (Experiment Comparison)")
    plt.grid(True)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()

    out_path = output_dir / "delay_over_time_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    fig_legend = plt.figure(figsize=(10, 2))
    fig_legend.legend(handles, legend_labels, loc="center", ncol=4)

    legend_path = output_dir / "delay_legend.png"
    fig_legend.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close(fig_legend)

    if show:
        plt.show()
    else:
        plt.close()

    return out_path, legend_path


def get_summary_inputs(args, output_dir: Path):
    file_paths = [Path(f).expanduser().resolve() for f in args.files]

    auto_roots = args.auto_find_csvs or []
    if auto_roots:
        discovered = discover_summary_csvs(
            auto_roots,
            name_filter=args.name_filter,
            sort_mode=args.sort,
            output_dir=output_dir,
        )
        existing = set(file_paths)
        for path in discovered:
            if path not in existing:
                file_paths.append(path)
                existing.add(path)

    if not file_paths and not args.skip_bar_plots:
        raise ValueError("No summary CSV files found or provided.")

    if args.labels:
        if len(args.labels) != len(file_paths):
            raise ValueError(
                f"You passed {len(args.labels)} labels for {len(file_paths)} summary CSV files. "
                "The counts must match."
            )
        labels = args.labels
    else:
        root_for_path_labels = Path(auto_roots[0]).expanduser().resolve() if len(auto_roots) == 1 else None
        labels = [make_label(p, args.label_mode, root=root_for_path_labels) for p in file_paths]

    return file_paths, labels


def get_delay_inputs(args):
    experiment_dirs = [Path(d).expanduser().resolve() for d in (args.delay_dirs or [])]

    auto_roots = args.auto_find_delay_dirs or []
    if auto_roots:
        discovered = discover_delay_dirs(
            auto_roots,
            name_filter=args.name_filter,
            sort_mode=args.sort,
        )
        existing = set(experiment_dirs)
        for path in discovered:
            if path not in existing:
                experiment_dirs.append(path)
                existing.add(path)

    if not experiment_dirs:
        return [], []

    if args.delay_labels:
        if len(args.delay_labels) != len(experiment_dirs):
            raise ValueError(
                f"You passed {len(args.delay_labels)} delay labels for "
                f"{len(experiment_dirs)} delay directories. The counts must match."
            )
        labels = args.delay_labels
    else:
        root_for_path_labels = Path(auto_roots[0]).expanduser().resolve() if len(auto_roots) == 1 else None
        labels = [make_label(p, args.label_mode, root=root_for_path_labels) for p in experiment_dirs]

    return experiment_dirs, labels


def run_bar_plot_workflow(args, output_dir: Path):
    if args.skip_bar_plots:
        return None, []

    file_paths, labels = get_summary_inputs(args, output_dir)
    summary = build_summary(file_paths, labels)
    metrics_to_plot = choose_metrics(summary, args.metrics, args.include_run_no)

    if not metrics_to_plot:
        raise ValueError("No metrics left to plot after filtering.")

    summary_csv = save_summary_csv(summary, output_dir)
    saved_plots = [plot_metric(summary, metric, output_dir, args.show) for metric in metrics_to_plot]

    return summary_csv, saved_plots


def run_delay_overlay_workflow(args, output_dir: Path):
    experiment_dirs, labels = get_delay_inputs(args)

    if not experiment_dirs:
        return None, None

    return plot_delay_overlay(
        experiment_dirs=experiment_dirs,
        labels=labels,
        output_dir=output_dir,
        nth=args.delay_nth,
        scatter_alpha=args.delay_scatter_alpha,
        window=args.delay_window,
        recursive=args.delay_recursive,
        show=args.show,
    )


def main():
    cli_args = parse_args()

    config = {}
    if cli_args.config:
        config = load_config_file(Path(cli_args.config).expanduser())

    args = merge_args_with_config(cli_args, config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv, saved_plots = run_bar_plot_workflow(args, output_dir)
    delay_plot, legend_plot = run_delay_overlay_workflow(args, output_dir)

    print("Comparison complete.")

    if summary_csv is not None:
        print(f"Summary CSV: {summary_csv}")
        print("Saved bar-plot files:")
        for path in saved_plots:
            print(path)

    if delay_plot is not None:
        print(f"Saved delay overlay plot: {delay_plot}")
        print(f"Saved delay legend: {legend_plot}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
