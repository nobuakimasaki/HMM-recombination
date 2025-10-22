#!/usr/bin/env python3
"""
Attach summary features from inferred lineage JSONs to optimization results:
- n_breakpoints: number of recombination breakpoints
- n_unique_lineages: number of distinct parental lineages
- parental_lineages: comma-separated list of parental lineages
- test_start, test_end: test window date range from filename
"""

import gzip
import json
from pathlib import Path
import pandas as pd
import glob
from Bio import SeqIO
import zstandard as zstd
from io import TextIOWrapper
from itertools import combinations

import sys, platform
print(sys.version)            # full version + build
print(platform.python_version())

# ========= Configuration ==========
data_dir = Path("../run-on-cluster-3/real-data-analysis/output/sliding_windows/inferred/")
json_pattern = str(data_dir / "inferred_lineages_*.json.gz")
csv_pattern  = str(data_dir / "optimization_results_*.csv.gz")
summary_dir = Path("summary_files")
summary_dir.mkdir(exist_ok=True)
output_path = summary_dir / "combined_optimization_results_with_summary.csv"
# ===================================

print("Looking for JSON files at:", json_pattern)
print("Looking for CSV files at:", csv_pattern)

json_files = sorted(glob.glob(json_pattern))
csv_files = sorted(glob.glob(csv_pattern))

print(f"Found {len(json_files)} JSON files:")
for f in json_files:
    print(" -", f)

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(" -", f)

# Build suffix maps
json_map = {
    Path(f).name.replace("inferred_lineages_", "").replace(".json.gz", ""): f
    for f in json_files
}
csv_map = {
    Path(f).name.replace("optimization_results_", "").replace(".csv.gz", ""): f
    for f in csv_files
}

# Intersect suffixes to find matching pairs only
common_suffixes = sorted(set(json_map) & set(csv_map))
paired_files = [(json_map[s], csv_map[s], s) for s in common_suffixes]

print(f"Found {len(paired_files)} matched JSON/CSV file pairs.")

combined = []

for json_fp, csv_fp, suffix in paired_files:
    parts = suffix.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format in suffix: {suffix}")
    test_start, test_end = parts[2], parts[3]

    with gzip.open(json_fp, "rt") as f:
        data = json.load(f)

    seq_ids = list(data.keys())

    n_breakpoints = []
    n_unique_lineages = []
    parental_lineages = []
    test_start_col = []
    test_end_col = []
    breakpoints_strs = []
    lineage_path_strs = []

    for blocks in data.values():
        lineages = set()
        breakpoints = 0
        bps = []
        path = []

        for block in blocks:
            if isinstance(block, list):
                if len(block) == 1:
                    path.append(block[0])
                    lineages.add(block[0])
                elif len(block) == 3:
                    path.append(block[0])
                    bps.append(str(block[1]))
                    path.append(block[2])
                    lineages.update([block[0], block[2]])

        # Remove consecutive duplicates in path
        dedup_path = [path[0]] if path else []
        for l in path[1:]:
            if l != dedup_path[-1]:
                dedup_path.append(l)

        n_breakpoints.append(len(bps))
        n_unique_lineages.append(len(lineages))
        parental_lineages.append(";".join(sorted(lineages)))
        test_start_col.append(test_start)
        test_end_col.append(test_end)
        breakpoints_strs.append(";".join(bps))
        lineage_path_strs.append(";".join(dedup_path))

    opt_df = pd.read_csv(csv_fp, compression="gzip")

    if "sequence_id" not in opt_df.columns:
        assert len(opt_df) == len(seq_ids), f"Mismatch in sequence count for {csv_fp}"
        opt_df.insert(0, "sequence_id", seq_ids)

    opt_df.insert(1, "test_start", test_start_col)
    opt_df.insert(2, "test_end", test_end_col)
    opt_df.insert(3, "n_breakpoints", n_breakpoints)
    opt_df.insert(4, "n_unique_lineages", n_unique_lineages)
    opt_df.insert(5, "parental_lineages", parental_lineages)

    opt_df["breakpoints"] = breakpoints_strs
    opt_df["lineage_path"] = lineage_path_strs

    combined.append(opt_df)

final_df = pd.concat(combined, ignore_index=True)
final_df.to_csv(output_path, index=False)
print(f"Saved combined CSV to: {output_path}")

# =======================
# Load and combine expected recombinant frequency files
# =======================

expected_dir = Path("../run-on-cluster-3/real-data-analysis/output/sliding_windows/expected_recombinant_freq")
expected_pattern = str(expected_dir / "expected_recombinants_*.csv.gz")

expected_files = sorted(glob.glob(expected_pattern))
print(f"Found {len(expected_files)} expected recombinant files.")

expected_dfs = []

for fp in expected_files:
    fname = Path(fp).name
    try:
        suffix = fname.replace("expected_recombinants_", "").replace(".csv.gz", "")
        test_start, test_end = suffix.split("_")
    except ValueError:
        raise ValueError(f"Unexpected file name format: {fname}")

    df = pd.read_csv(fp, compression="gzip")
    df["test_start"] = test_start
    df["test_end"] = test_end

    expected_dfs.append(df)

expected_combined = pd.concat(expected_dfs, ignore_index=True)
expected_out_path = summary_dir / "combined_expected_recombinants.csv"
expected_combined.to_csv(expected_out_path, index=False)
print(f"Saved combined expected recombinants to: {expected_out_path}")

# =======================
# Calculate Hamming distances between lineages (ignoring Ns)
# =======================

fasta_path = "../data/pango-consensus-sequences_genome-nuc.fasta.zst"
print(f"Reading consensus sequences from: {fasta_path}")

sequences = {}
with open(fasta_path, "rb") as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        text_stream = TextIOWrapper(reader, encoding='utf-8')
        for record in SeqIO.parse(text_stream, "fasta"):
            seq = str(record.seq).upper()
            sequences[record.id] = seq

sequences = {k: v for k, v in sequences.items() if set(v) != {"N"} and len(v) > 0}
print(f"Loaded {len(sequences)} sequences with valid content.")

expected_combined[["Lineage_1", "Lineage_2"]] = expected_combined[["Lineage_A", "Lineage_B"]].apply(
    sorted, axis=1, result_type="expand"
)

unique_pairs = expected_combined[["Lineage_1", "Lineage_2"]].drop_duplicates()

def hamming_distance_ignore_N(seq1, seq2):
    return sum(nt1 != nt2 for nt1, nt2 in zip(seq1, seq2) if nt1 != "N" and nt2 != "N")

records = []
for _, row in unique_pairs.iterrows():
    lin1, lin2 = row["Lineage_1"], row["Lineage_2"]
    if lin1 in sequences and lin2 in sequences:
        dist = hamming_distance_ignore_N(sequences[lin1], sequences[lin2])
        records.append((lin1, lin2, dist))
    else:
        print(f"Skipping missing sequence(s): {lin1}, {lin2}")

hamming_df = pd.DataFrame(records, columns=["Lineage_1", "Lineage_2", "Hamming_Distance"])
hamming_out_path = summary_dir / "hamming_distances.csv"
hamming_df.to_csv(hamming_out_path, index=False)
print(f"Saved filtered Hamming distances to: {hamming_out_path}")
