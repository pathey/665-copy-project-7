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
        "--graphs",
        nargs="+",
        choices=["bar_plots", "delay_overlay", "energy_consumption"],
        default=None,
        help="Graph workflows to run. Use config key graphs for the same setting."
    )
    parser.add_argument(
        "--graph-group-size",
        type=int,
        default=None,
        help="Maximum number of experiments per generated graph. Values over this split every graph type into multiple PNGs."
    )
    parser.add_argument(
        "--bar-group-size",
        type=int,
        default=None,
        help="Deprecated alias for --graph-group-size."
    )
    parser.add_argument(
        "--energy-metrics",
        nargs="+",
        default=None,
        help="Metric column names to use for the energy consumption graph. If omitted, energy-like columns are auto-detected."
    )
    parser.add_argument(
        "--energy-dirs",
        nargs="+",
        help="Experiment folders that contain Flow_Node*.csv files for energy-over-time plots"
    )
    parser.add_argument(
        "--energy-labels",
        nargs="+",
        help="Custom labels for energy folders, in the same order as --energy-dirs"
    )
    parser.add_argument(
        "--energy-recursive",
        action="store_true",
        default=None,
        help="Recursively search each energy directory for Flow_Node*.csv files"
    )
    parser.add_argument(
        "--energy-time-column",
        choices=["First Tx At (in s)", "Generation Time (in s)", "Reception Time (s)"],
        default=None,
        help="Time column to use for energy-over-time plots"
    )
    parser.add_argument(
        "--energy-rate-column",
        choices=["Min. Data Rate Along Route (in Mbps)", "Data Rate at Flow Src (in Mbps)"],
        default=None,
        help="Data-rate column used to estimate transmission energy"
    )
    parser.add_argument(
        "--energy-rolling-window",
        type=int,
        default=None,
        help="Rolling-average window for energy-over-time plots"
    )
    parser.add_argument(
        "--energy-nth",
        type=int,
        default=None,
        help="Plot every nth raw energy point"
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
        "--auto-find-energy-dirs",
        nargs="+",
        help="One or more root directories to recursively search for experiment folders containing Flow_Node*.csv for energy plots"
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
        "graphs": ["bar_plots", "delay_overlay"],
        "graph_group_size": None,
        "bar_group_size": None,
        "energy_metrics": None,
        "energy_dirs": None,
        "energy_labels": None,
        "energy_recursive": False,
        "energy_time_column": "First Tx At (in s)",
        "energy_rate_column": "Min. Data Rate Along Route (in Mbps)",
        "energy_rolling_window": 50,
        "energy_nth": 1,
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
        "auto_find_energy_dirs": None,
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

    if merged.graph_group_size is None:
        merged.graph_group_size = merged.bar_group_size if merged.bar_group_size is not None else 0

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


def normalize_graphs(graphs):
    if graphs is None:
        return {"bar_plots", "delay_overlay"}
    if isinstance(graphs, str):
        graphs = [graphs]
    return set(graphs)


def split_sequence(items, labels, group_size: int):
    if len(items) != len(labels):
        raise ValueError(f"Cannot split graph inputs: {len(items)} items but {len(labels)} labels.")

    if group_size is None or group_size <= 0 or len(items) <= group_size:
        yield None, items, labels
        return

    for start in range(0, len(items), group_size):
        stop = min(start + group_size, len(items))
        yield f"part_{start // group_size + 1:02d}", items[start:stop], labels[start:stop]


def split_summary(summary: pd.DataFrame, group_size: int):
    for suffix, _, _ in split_sequence(list(summary.index), list(summary.index), group_size):
        if suffix is None:
            yield None, summary
        else:
            part_no = int(suffix.rsplit("_", 1)[1]) - 1
            start = part_no * group_size
            stop = min(start + group_size, len(summary))
            yield suffix, summary.iloc[start:stop]


def detect_energy_metrics(summary: pd.DataFrame, requested_metrics):
    available = list(summary.columns)

    if requested_metrics:
        missing = [m for m in requested_metrics if m not in available]
        if missing:
            raise ValueError(
                f"Requested energy metric(s) not found: {', '.join(missing)}\n"
                f"Available metrics are: {', '.join(available)}"
            )
        return requested_metrics

    keywords = ("energy", "consumption", "joule", "power")
    return [m for m in available if any(k in str(m).lower() for k in keywords)]


def save_summary_csv(summary: pd.DataFrame, output_dir: Path):
    out_csv = output_dir / "summary_comparison.csv"
    summary.to_csv(out_csv)
    return out_csv


def plot_metric(summary: pd.DataFrame, metric: str, output_dir: Path, show: bool, filename_suffix=None):
    values = summary[metric]

    width = max(10, min(18, 0.65 * max(1, len(values))))
    plt.figure(figsize=(width, 6))
    ax = values.plot(kind="bar")

    title = metric if filename_suffix is None else f"{metric} ({filename_suffix.replace('_', ' ')})"
    plt.title(title)
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

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = output_dir / f"{sanitize_filename(metric)}{suffix}.png"
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
    filename_suffix=None,
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
    title = "Packet Delay over Time (Experiment Comparison)"
    if filename_suffix:
        title = f"{title} ({filename_suffix.replace('_', ' ')})"
    plt.title(title)
    plt.grid(True)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = output_dir / f"delay_over_time_comparison{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    fig_legend = plt.figure(figsize=(10, 2))
    fig_legend.legend(handles, legend_labels, loc="center", ncol=4)

    legend_path = output_dir / f"delay_legend{suffix}.png"
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
    graphs = normalize_graphs(args.graphs)
    if args.skip_bar_plots or "bar_plots" not in graphs:
        return None, []

    file_paths, labels = get_summary_inputs(args, output_dir)
    summary = build_summary(file_paths, labels)
    metrics_to_plot = choose_metrics(summary, args.metrics, args.include_run_no)

    if not metrics_to_plot:
        raise ValueError("No metrics left to plot after filtering.")

    summary_csv = save_summary_csv(summary, output_dir)
    saved_plots = []
    for metric in metrics_to_plot:
        for suffix, summary_chunk in split_summary(summary, args.graph_group_size):
            saved_plots.append(plot_metric(summary_chunk, metric, output_dir, args.show, filename_suffix=suffix))

    return summary_csv, saved_plots



def parse_power_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, (int, float)):
        val = float(value)
        return [val] if val == val and val != float("inf") and val != float("-inf") else []

    numbers = []
    for item in str(value).split(","):
        item = item.strip().strip("[]()")
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError:
            continue
        if parsed == parsed and parsed != float("inf") and parsed != float("-inf"):
            numbers.append(parsed)
    return numbers


def get_energy_inputs(args):
    experiment_dirs = [Path(d).expanduser().resolve() for d in (args.energy_dirs or [])]

    auto_roots = args.auto_find_energy_dirs or []
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

    if args.energy_labels:
        if len(args.energy_labels) != len(experiment_dirs):
            raise ValueError(
                f"You passed {len(args.energy_labels)} energy labels for "
                f"{len(experiment_dirs)} energy directories. The counts must match."
            )
        labels = args.energy_labels
    else:
        root_for_path_labels = Path(auto_roots[0]).expanduser().resolve() if len(auto_roots) == 1 else None
        labels = [make_label(p, args.label_mode, root=root_for_path_labels) for p in experiment_dirs]

    return experiment_dirs, labels


def load_energy_points_for_experiment(experiment_dir: Path, recursive: bool, time_column: str, rate_column: str):
    flow_csvs = find_flow_csvs(experiment_dir, recursive=recursive)

    if not flow_csvs:
        raise ValueError(f"No Flow_Node*.csv files found in {experiment_dir}")

    all_frames = []
    for csv_path in flow_csvs:
        df = pd.read_csv(csv_path)

        required = {"Pkt ID", "Packet Size (in bytes)", "Tx Power (in watt)", rate_column}
        if time_column == "Reception Time (s)":
            required.update({"Generation Time (in s)", "Delay (in s)"})
        else:
            required.add(time_column)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required column(s) {sorted(missing)} in flow CSV: {csv_path}")

        working = df.copy()
        working["Packet Size (in bytes)"] = pd.to_numeric(working["Packet Size (in bytes)"], errors="coerce")
        working[rate_column] = pd.to_numeric(working[rate_column], errors="coerce")

        if time_column == "Reception Time (s)":
            working["Generation Time (in s)"] = pd.to_numeric(working["Generation Time (in s)"], errors="coerce")
            working["Delay (in s)"] = pd.to_numeric(working["Delay (in s)"], errors="coerce")
            working["Energy Time (s)"] = working["Generation Time (in s)"] + working["Delay (in s)"]
        else:
            working["Energy Time (s)"] = pd.to_numeric(working[time_column], errors="coerce")

        working["Tx Power List (W)"] = working["Tx Power (in watt)"].apply(parse_power_list)
        working["Total Tx Power Across Route (W)"] = working["Tx Power List (W)"].apply(sum)
        working["Tx Duration Estimate (s)"] = working["Packet Size (in bytes)"] * 8.0 / (working[rate_column] * 1_000_000.0)
        working["Energy Estimate (J)"] = working["Total Tx Power Across Route (W)"] * working["Tx Duration Estimate (s)"]

        keep = working[
            working["Energy Time (s)"].notna()
            & working["Energy Estimate (J)"].notna()
            & (working["Energy Estimate (J)"] >= 0)
            & working["Tx Duration Estimate (s)"].notna()
            & (working["Tx Duration Estimate (s)"] > 0)
        ].copy()

        if keep.empty:
            continue

        all_frames.append(keep[["Pkt ID", "Energy Time (s)", "Energy Estimate (J)", "Total Tx Power Across Route (W)", "Tx Duration Estimate (s)"]])

    if not all_frames:
        raise ValueError(f"No valid energy rows found in {experiment_dir}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("Energy Time (s)").reset_index(drop=True)
    combined["Cumulative Energy Estimate (J)"] = combined["Energy Estimate (J)"].cumsum()
    return combined


def plot_energy_overlay(
    experiment_dirs,
    labels,
    output_dir: Path,
    recursive: bool,
    time_column: str,
    rate_column: str,
    rolling_window: int,
    nth: int,
    show: bool,
    filename_suffix=None,
):
    loaded = []
    summary_rows = []

    for experiment_dir, label in zip(experiment_dirs, labels):
        df = load_energy_points_for_experiment(experiment_dir, recursive, time_column, rate_column)
        df["Experiment"] = label
        loaded.append((label, df))
        summary_rows.append({
            "Experiment": label,
            "Packets": len(df),
            "Total Energy Estimate (J)": df["Energy Estimate (J)"].sum(),
            "Mean Packet Energy Estimate (J)": df["Energy Estimate (J)"].mean(),
            "Max Packet Energy Estimate (J)": df["Energy Estimate (J)"].max(),
            "Mean Total Tx Power Across Route (W)": df["Total Tx Power Across Route (W)"].mean(),
            "Mean Tx Duration Estimate (s)": df["Tx Duration Estimate (s)"].mean(),
            "Start Time (s)": df["Energy Time (s)"].min(),
            "End Time (s)": df["Energy Time (s)"].max(),
        })

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    summary_csv = output_dir / f"energy_summary{suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    saved = []

    def save(kind, y_col, title, y_label, scatter=False):
        plt.figure(figsize=(11, 6))
        for label, df in loaded:
            if scatter:
                sampled = df.iloc[::max(1, nth), :]
                plt.scatter(sampled["Energy Time (s)"], sampled[y_col], s=10, alpha=0.45, label=label)
            else:
                plt.plot(df["Energy Time (s)"], df[y_col], linewidth=2, label=label)
        full_title = title if not filename_suffix else f"{title} ({filename_suffix.replace('_', ' ')})"
        plt.title(full_title)
        plt.xlabel("Time (s)")
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_path = output_dir / f"{kind}{suffix}.png"
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        saved.append(out_path)
        if show:
            plt.show()
        else:
            plt.close()

    save("cumulative_energy_over_time", "Cumulative Energy Estimate (J)", "Estimated Cumulative Transmission Energy Over Time", "Cumulative Energy Estimate (J)")
    save("packet_energy_over_time", "Energy Estimate (J)", "Estimated Per-Packet Transmission Energy Over Time", "Energy Estimate Per Packet (J)", scatter=True)

    if rolling_window and rolling_window > 1:
        for _, df in loaded:
            df["Rolling Energy Estimate (J)"] = df["Energy Estimate (J)"].rolling(window=rolling_window, min_periods=1).mean()
        save("rolling_packet_energy_over_time", "Rolling Energy Estimate (J)", f"Rolling Mean Packet Energy Over Time (window={rolling_window})", "Rolling Mean Packet Energy (J)")

    return summary_csv, saved

def run_energy_consumption_workflow(args, output_dir: Path):
    graphs = normalize_graphs(args.graphs)
    if "energy_consumption" not in graphs:
        return [], []

    experiment_dirs, labels = get_energy_inputs(args)

    # Preferred path: use Flow_Node*.csv files and plot energy over time for all selected experiments.
    if experiment_dirs:
        saved_csvs = []
        saved_plots = []
        for suffix, dirs_chunk, labels_chunk in split_sequence(experiment_dirs, labels, args.graph_group_size):
            summary_csv, plots = plot_energy_overlay(
                experiment_dirs=dirs_chunk,
                labels=labels_chunk,
                output_dir=output_dir,
                recursive=args.energy_recursive,
                time_column=args.energy_time_column,
                rate_column=args.energy_rate_column,
                rolling_window=args.energy_rolling_window,
                nth=args.energy_nth,
                show=args.show,
                filename_suffix=suffix,
            )
            saved_csvs.append(summary_csv)
            saved_plots.extend(plots)
        return saved_csvs, saved_plots

    # Fallback path: if no energy dirs are provided, plot energy-like columns from all_stats files.
    file_paths, labels = get_summary_inputs(args, output_dir)
    summary = build_summary(file_paths, labels)
    metrics_to_plot = detect_energy_metrics(summary, args.energy_metrics)

    if not metrics_to_plot:
        print(
            "Warning: energy_consumption was requested, but no energy directories or energy-like summary metrics were found. "
            "Set energy_dirs in the config to the experiment folders containing Flow_Node*.csv files.",
            file=sys.stderr,
        )
        return [], []

    saved_plots = []
    for metric in metrics_to_plot:
        safe_metric = metric if "energy" in str(metric).lower() else f"Energy Consumption - {metric}"
        for suffix, summary_chunk in split_summary(summary, args.graph_group_size):
            saved_plots.append(plot_metric(summary_chunk.rename(columns={metric: safe_metric}), safe_metric, output_dir, args.show, filename_suffix=suffix))
    return [], saved_plots

def run_delay_overlay_workflow(args, output_dir: Path):
    graphs = normalize_graphs(args.graphs)
    if "delay_overlay" not in graphs:
        return None, None

    experiment_dirs, labels = get_delay_inputs(args)

    if not experiment_dirs:
        return None, None

    saved = []
    legends = []
    for suffix, dirs_chunk, labels_chunk in split_sequence(experiment_dirs, labels, args.graph_group_size):
        delay_plot, legend_plot = plot_delay_overlay(
            experiment_dirs=dirs_chunk,
            labels=labels_chunk,
            output_dir=output_dir,
            nth=args.delay_nth,
            scatter_alpha=args.delay_scatter_alpha,
            window=args.delay_window,
            recursive=args.delay_recursive,
            show=args.show,
            filename_suffix=suffix,
        )
        saved.append(delay_plot)
        legends.append(legend_plot)

    return saved, legends


def main():
    cli_args = parse_args()

    config = {}
    if cli_args.config:
        config = load_config_file(Path(cli_args.config).expanduser())

    args = merge_args_with_config(cli_args, config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv, saved_plots = run_bar_plot_workflow(args, output_dir)
    energy_csvs, energy_plots = run_energy_consumption_workflow(args, output_dir)
    delay_plots, legend_plots = run_delay_overlay_workflow(args, output_dir)

    print("Comparison complete.")

    if summary_csv is not None:
        print(f"Summary CSV: {summary_csv}")
        print("Saved bar-plot files:")
        for path in saved_plots:
            print(path)

    if energy_csvs:
        print("Saved energy summary CSV files:")
        for path in energy_csvs:
            print(path)

    if energy_plots:
        print("Saved energy-consumption plot files:")
        for path in energy_plots:
            print(path)

    if delay_plots:
        print("Saved delay overlay plot files:")
        for path in delay_plots:
            print(path)
        print("Saved delay legend files:")
        for path in legend_plots:
            print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
