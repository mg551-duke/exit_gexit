#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# --- Mode labels (includes 6 and 7) ---
MODE_LABELS = {
    "1": "Peeling",
    "2": "Inactivation",
    "3": "Peeling+Stab",
    "4": "Stab+Inact",
    "6": "Hard guessing",
    "7": "Stab+Hard guessing",
}

# Keep a single source of truth for which modes to plot
PLOT_MODES = sorted(MODE_LABELS.keys(), key=int)


def iter_json_files(folder: Path):
    # adjust this if you want recursive: folder.rglob("*.json")
    yield from sorted(folder.glob("*.json"))


def load_and_aggregate(folder: Path):
    """
    Returns:
      agg[mode][p] = dict(logic=..., nonconv=..., runs=...)
    Aggregation rule:
      sums logic/nonconv/runs across files (assumes files cover disjoint seeds/runs per p).
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"logic": 0, "nonconv": 0, "runs": 0}))
    files = list(iter_json_files(folder))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    for fp in files:
        with fp.open("r") as f:
            data = json.load(f)

        results = data.get("results", {})
        for mode, entries in results.items():
            # Only keep the modes we know how to label/plot
            if mode not in MODE_LABELS:
                continue

            for e in entries:
                p = round(float(e["p"]), 6)
                agg[mode][p]["logic"] += int(e.get("logic", 0))
                agg[mode][p]["nonconv"] += int(e.get("nonconv", 0))
                agg[mode][p]["runs"] += int(e.get("runs", 0))

    return agg


def mode_curve(agg_for_mode: dict):
    """
    Input:
      agg_for_mode: dict[p] -> {logic, nonconv, runs}
    Output:
      ps_sorted, err_sorted
    """
    ps = sorted(agg_for_mode.keys())
    errs = []
    for p in ps:
        logic = agg_for_mode[p]["logic"]
        nonconv = agg_for_mode[p]["nonconv"]
        runs = agg_for_mode[p]["runs"]
        errs.append(float("nan") if runs <= 0 else (logic + nonconv) / runs)
    return ps, errs


def plot_datasets(datasets, title, logy=True):
    # Use fig/ax API so legend placement is reliable
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Unique line styles per MODE (no duplicates) ---
    # These are matplotlib "linestyle" tuples (offset, on_off_seq) for dash patterns.
    linestyles = {
        "1": (0, ()),              # solid
        "2": (0, (6, 2)),           # long dash
        "3": (0, (2, 2)),           # short dash
        "4": (0, (1, 2)),           # dotted
        "6": (0, (6, 2, 1, 2)),     # dash-dot (long)
        "7": (0, (2, 2, 1, 2)),     # dash-dot (short)
    }

    # Marker is used to distinguish datasets (folders)
    markers = ["o", "s", "^", "D", "v", "P", "X"]  # one per dataset (cycled)

    for i, (label, folder) in enumerate(datasets):
        agg = load_and_aggregate(folder)
        marker = markers[i % len(markers)]

        for mode in PLOT_MODES:
            if mode not in agg:
                continue
            ps, errs = mode_curve(agg[mode])
            if not ps:
                continue

            ax.plot(
                ps,
                errs,
                marker=marker,
                linestyle=linestyles.get(mode, (0, ())),  # fallback to solid
                label=f"{label} {MODE_LABELS[mode]}",
            )

    ax.set_xlabel("Channel parameter p")
    ax.set_ylabel("Logical error rate ( (logic + nonconv) / runs )")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # --- Legend OUTSIDE the axes on the right ---
    # Make space on the right for the legend, then anchor it outside.
    fig.subplots_adjust(right=0.72)  # increase/decrease to give legend more/less room
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize="small",
        frameon=True,
    )

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=".",
        help="Path to the folder containing Sum_total_B1 / Sum_total_HGP2025 / etc.",
    )
    ap.add_argument(
        "--folders",
        nargs="*",
        default=[
            "Sum_total_B1",
            #"Sum_total_HGP2025",
            #"Sum_total_Surface11",
            #"Sum_total_Surface13",
            #"Sum_total_BB_n108_k8_l9_m6_Ax3_y1_y2_By3_x1_x2",
            #"Sum_total_BB_n144_k12_l12_m6_Ax3_y1_y2_By3_x1_x2.npz",
            #"Sum_total_BB_n288_k12_l12_m12_Ax3_y2_y7_By3_x1_x2.npz",
        ],
        help="Which Sum_total_* folders to include (relative to --base)",
    )
    ap.add_argument("--title", default="Decoder Comparison", help="Plot title")
    ap.add_argument("--no-logy", action="store_true", help="Disable log-scale y axis")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    datasets = []
    for name in args.folders:
        folder = base / name
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        # nicer label: strip prefix if present
        label = name.replace("Sum_total_", "")
        datasets.append((label, folder))

    plot_datasets(datasets, title=args.title, logy=(not args.no_logy))


if __name__ == "__main__":
    main()


