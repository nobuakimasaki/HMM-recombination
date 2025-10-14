import json
import glob
import gzip
import os
from collections import defaultdict, Counter
from typing import List, Union, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2

# ------------------------------
# Entropy Utilities
# ------------------------------

def shannon_entropy(proportions: List[float]) -> float:
    """Compute Shannon entropy from a list of proportions."""
    return -sum(p * log2(p) for p in proportions if p > 0)


def expand_lineage_path(path: List[List[Union[str, int]]], seq_length: int = 29903) -> List[str]:
    """Expand compressed lineage path representation into full-length array."""
    lineage = [None] * seq_length
    cur_pos = 0

    for segment in path:
        if len(segment) == 1:
            lineage[cur_pos:] = [segment[0]] * (seq_length - cur_pos)
            break
        elif len(segment) == 3:
            left, breakpoint, right = segment
            lineage[cur_pos:breakpoint] = [left] * (breakpoint - cur_pos)
            cur_pos = breakpoint

    if cur_pos < seq_length and len(path[-1]) == 3:
        lineage[cur_pos:] = [path[-1][2]] * (seq_length - cur_pos)

    return lineage


def extract_lineage_arrays(data: Dict, seq_length: int = 29903) -> Dict[str, List[str]]:
    """Extract full-length lineage arrays for recombinant sequences."""
    lineage_arrays = {}
    for seq_id, path in data.items():
        if not path or (len(path) == 1 and len(path[0]) == 1):
            continue  # Skip truly non-recombinant (e.g., [['AY.4']])
        expanded = expand_lineage_path(path, seq_length)
        if len(set(expanded)) > 1:
            lineage_arrays[seq_id] = expanded
    return lineage_arrays


def group_by_lineage_pair(lineage_arrays: Dict[str, List[str]]) -> Dict[Tuple[str, str], List[List[str]]]:
    """Group lineage paths by lineage pairs (ignoring direction)."""
    pair_to_paths = defaultdict(list)
    for lineage_array in lineage_arrays.values():
        unique_lineages = sorted(set(lineage_array))
        if len(unique_lineages) == 2:
            pair_to_paths[tuple(unique_lineages)].append(lineage_array)
    return pair_to_paths


def compute_entropy_matrix(seqs: List[List[str]]) -> pd.Series:
    """Compute positional Shannon entropy across aligned lineage paths."""
    arr = np.array(seqs)
    entropy_by_pos = {}
    for j in range(arr.shape[1]):
        freqs = Counter(arr[:, j])
        total = sum(freqs.values())
        proportions = [count / total for count in freqs.values()]
        entropy_by_pos[j] = shannon_entropy(proportions)
    return pd.Series(entropy_by_pos)


def summarize_entropy_from_json(json_path: str, seq_length: int = 29903) -> pd.DataFrame:
    """Compute entropy summaries from a single inferred JSON."""
    open_func = gzip.open if json_path.endswith(".gz") else open
    with open_func(json_path, 'rt') as f:
        data = json.load(f)

    lineage_arrays = extract_lineage_arrays(data, seq_length)
    pair_to_paths = group_by_lineage_pair(lineage_arrays)

    summary = []
    for (a, b), paths in pair_to_paths.items():
        entropy_series = compute_entropy_matrix(paths)
        summary.append({
            "Lineage_1": a,
            "Lineage_2": b,
            "Mean_Entropy": entropy_series.mean(),
            "Max_Entropy": entropy_series.max(),
            "Positions_Over_0.5": (entropy_series > 0.5).sum()
        })
    return pd.DataFrame(summary)


def summarize_entropy_from_multiple_json(json_pattern: str, seq_length: int = 29903) -> pd.DataFrame:
    """Aggregate entropy summaries from multiple inferred JSONs."""
    all_lineage_arrays = {}

    for path in glob.glob(json_pattern):
        open_func = gzip.open if path.endswith(".gz") else open
        with open_func(path, 'rt') as f:
            data = json.load(f)
        lineage_arrays = extract_lineage_arrays(data, seq_length)
        all_lineage_arrays.update(lineage_arrays)

    pair_to_paths = group_by_lineage_pair(all_lineage_arrays)

    summary = []
    for (a, b), paths in pair_to_paths.items():
        entropy_series = compute_entropy_matrix(paths)
        summary.append({
            "Lineage_1": a,
            "Lineage_2": b,
            "Mean_Entropy": entropy_series.mean(),
            "Max_Entropy": entropy_series.max(),
            "Positions_Over_0.5": (entropy_series > 0.5).sum()
        })
    return pd.DataFrame(summary)

# ------------------------------
# Plotting
# ------------------------------

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_lineage_pair_paths(
    json_pattern: str,
    lineage_pair: Tuple[str, str],
    seq_length: int = 29903,
    output_dir: str = ".",
    caption: str = None,
    mutation_positions: List[int] = None
):
    """Plot recombinant lineage paths for a given lineage pair and save as PNG with custom legend."""
    all_lineage_arrays = {}

    for path in glob.glob(json_pattern):
        open_func = gzip.open if path.endswith(".gz") else open
        with open_func(path, 'rt') as f:
            data = json.load(f)
        lineage_arrays = extract_lineage_arrays(data, seq_length)
        all_lineage_arrays.update(lineage_arrays)

    lineage_1, lineage_2 = sorted(lineage_pair)
    selected_arrays = [
        arr for arr in all_lineage_arrays.values()
        if set(arr) == {lineage_1, lineage_2}
    ]

    if not selected_arrays:
        print(f"No sequences found for pair {lineage_pair}")
        return

    # Convert to numeric and plot
    arr = np.array(selected_arrays)
    lineage_codes = {lineage_1: 0, lineage_2: 1}
    numeric_arr = np.vectorize(lineage_codes.get)(arr)

    # Define custom colormap and legend
    lineage_colors = {lineage_1: "blue", lineage_2: "red"}
    cmap = mcolors.ListedColormap([lineage_colors[lineage_1], lineage_colors[lineage_2]])

    plt.figure(figsize=(min(seq_length // 50, 20), min(len(numeric_arr) // 2 + 2, 20)))
    sns.heatmap(
        numeric_arr,
        cmap=cmap,
        cbar=False,
        yticklabels=False,
        xticklabels=False
    )

    if mutation_positions:
        for pos in mutation_positions:
            plt.axvline(pos, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.title(f"Recombinant Paths: {lineage_1} ↔ {lineage_2}")
    plt.xlabel("Genome Position")
    plt.ylabel("Sequence Index")

    # Add custom legend
    legend_handles = [
        mpatches.Patch(color=lineage_colors[lineage_1], label=lineage_1),
        mpatches.Patch(color=lineage_colors[lineage_2], label=lineage_2)
    ]
    plt.legend(
        handles=legend_handles,
        title="Lineages",
        loc="upper right",
        bbox_to_anchor=(1.15, 1)
    )

    # Optional caption
    if caption:
        plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', fontsize=10)

    plt.tight_layout()
    filename = f"{lineage_1}_{lineage_2}_paths.png".replace("/", "_")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
