import json
from collections import Counter
# --- Stats + Plots ---
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_PATH = "../data/usher.json"
X_PREFIX = "X"              # disregard names starting with this (but still traverse)
SKIP_SUBSTR = "internal"    # disregard names containing this (case-insensitive)
SKIP_PREFIXES = ("NODE",)   # disregard names starting with any of these (case-insensitive)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_node_by_name(node, target):
    if node.get("name") == target:
        return node
    for child in node.get("children", []):
        hit = find_node_by_name(child, target)
        if hit is not None:
            return hit
    return None

def should_disregard(name: str, x_prefix: str, skip_substr: str, skip_prefixes: tuple) -> bool:
    if not isinstance(name, str):
        return False
    nlow = name.lower()
    if name.startswith(x_prefix):
        return True
    if any(name.upper().startswith(p.upper()) for p in skip_prefixes):
        return True
    if skip_substr and (skip_substr.lower() in nlow):
        return True
    return False

def count_from_rec_parent_traverse_through_skips(
    root, x_prefix="X", skip_substr="internal", skip_prefixes=("NODE",), include_start=True
):
    """
    Start at 'rec_parent'. Disregard nodes whose names:
      - start with x_prefix (e.g., 'X...'), OR
      - start with any in skip_prefixes (e.g., 'NODE...'), OR
      - contain skip_substr (case-insensitive, e.g., 'internal').

    Disregarded nodes are NOT counted but ARE traversed.
    """
    tree_root = root.get("tree", root)
    start = find_node_by_name(tree_root, "rec_parent")
    if start is None:
        raise ValueError("Could not find node named 'rec_parent'")

    per_branch = []
    total = 0
    visited = set()

    skipped_nodes = 0
    counted_nodes = 0
    visited_nodes = 0

    def dfs(n, is_start=False):
        nonlocal total, skipped_nodes, counted_nodes, visited_nodes
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)
        visited_nodes += 1

        name = n.get("name", "") or "<unnamed>"
        skip_here = should_disregard(name, x_prefix, skip_substr, skip_prefixes)

        # Count nuc mutations on incoming branch unless:
        #  - this node is disregarded, OR
        #  - this is the starting node and include_start=False
        if (not skip_here) and not (is_start and not include_start):
            nuc = (
                n.get("branch_attrs", {})
                 .get("mutations", {})
                 .get("nuc", [])
            )
            if nuc is None:
                nuc = []
            elif isinstance(nuc, dict):
                nuc = list(nuc.values())
            cnt = len(nuc)
            per_branch.append((name, cnt))
            total += cnt
            counted_nodes += 1
        else:
            skipped_nodes += 1

        # Always traverse children
        for child in n.get("children", []):
            dfs(child, is_start=False)

    dfs(start, is_start=True)
    return per_branch, total, {
        "visited_nodes": visited_nodes,
        "counted_nodes": counted_nodes,
        "disregarded_nodes": skipped_nodes
    }

if __name__ == "__main__":
    data = load_json(INPUT_PATH)
    per_branch, grand_total, stats = count_from_rec_parent_traverse_through_skips(
        data,
        x_prefix=X_PREFIX,
        skip_substr=SKIP_SUBSTR,
        skip_prefixes=SKIP_PREFIXES,
        include_start=True
    )

    print(f"Total nucleotide mutations (start at rec_parent; disregard X*, 'NODE*', and names containing '{SKIP_SUBSTR}'):", grand_total, "\n")
    print(f"Visited nodes: {stats['visited_nodes']}")
    print(f"Counted nodes: {stats['counted_nodes']}")
    print(f"Disregarded (but traversed) nodes: {stats['disregarded_nodes']}\n")

    # --- Print ALL counted branches (traversal order) ---
    print("All counted branches (traversal order):")
    for name, cnt in per_branch:
        print(f"{name}\t{cnt}")

# just the counts, in traversal order
counts = [cnt for _, cnt in per_branch]

# filter out zeros
counts_nz = [int(c) for c in counts if int(c) != 0]

out_dir = os.path.join(".", "output", "mutation_counts")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "mutation_count")

with open(out_path, "w") as f:
    for c in counts_nz:
        f.write(f"{c}\n")

print(f"Wrote {len(counts_nz)} nonzero branch mutation counts to {out_path}")

n = len(counts_nz)

if n == 0:
    print("\nNo counted branches. Nothing to show.")
else:
    counts_arr = np.array(counts_nz, dtype=float)

    mean = counts_arr.mean()
    median = np.median(counts_arr)
    q1 = np.percentile(counts_arr, 25)
    q3 = np.percentile(counts_arr, 75)
    iqr = q3 - q1
    cmin = counts_arr.min()
    cmax = counts_arr.max()
    perc95 = np.percentile(counts_arr, 95)

    print("\n--- Mutation count summary (per counted branch) ---")
    print(f"n = {n}")
    print(f"mean = {mean:.3f}")
    print(f"median = {median:.3f}")
    print(f"Q1 = {q1:.3f}, Q3 = {q3:.3f}, IQR = {iqr:.3f}")
    print(f"min = {cmin:.0f}, max = {cmax:.0f}")
    print(f"95th percentile = {perc95:.3f}")

    # # Make an output dir next to the input by default
    # out_dir = os.path.join(os.path.dirname(INPUT_PATH), "plots")
    # os.makedirs(out_dir, exist_ok=True)

    # # 1) Histogram (binwidth=1 works well for counts)
    # plt.figure()
    # bins = np.arange(cmin - 0.5, cmax + 1.5, 1)
    # plt.hist(counts_arr, bins=bins)
    # plt.xlabel("Substitution count per branch")
    # plt.ylabel("Frequency")
    # plt.title("Empirical distribution of branch-wise substitution counts")
    # hist_path = os.path.join(out_dir, "branch_counts_hist.png")
    # plt.tight_layout()
    # plt.savefig(hist_path, dpi=200)
    # plt.close()
    # print(f"Saved histogram: {hist_path}")

    # # 2) ECDF
    # plt.figure()
    # xs = np.sort(counts_arr)
    # ys = np.arange(1, n + 1) / n
    # plt.step(xs, ys, where="post")
    # plt.xlabel("Substitution count per branch")
    # plt.ylabel("ECDF")
    # plt.title("ECDF of branch-wise substitution counts")
    # ecdf_path = os.path.join(out_dir, "branch_counts_ecdf.png")
    # plt.tight_layout()
    # plt.savefig(ecdf_path, dpi=200)
    # plt.close()
    # print(f"Saved ECDF: {ecdf_path}")

    # # Optional: export a TSV of (name, count) for downstream use
    # tsv_path = os.path.join(out_dir, "branch_counts.tsv")
    # with open(tsv_path, "w") as f:
    #     f.write("name\tcount\n")
    #     for name, cnt in per_branch:
    #         f.write(f"{name}\t{cnt}\n")
    # print(f"Saved counts table: {tsv_path}")
