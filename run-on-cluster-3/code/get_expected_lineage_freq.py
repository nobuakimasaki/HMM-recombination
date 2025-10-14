#!/usr/bin/env python3
"""
get_expected_lineage_freq.py

Compute
  (i)  expected recombinant frequencies 2 p_A p_B for every unordered
       pair of collapsed Pango lineages   →  --out <pairwise>.csv.gz
  (ii) per-lineage frequencies            →  --lineage_freq_out <single>.csv.gz
"""
import argparse
from itertools import combinations
import pandas as pd

# ──────────── CLI ────────────
parser = argparse.ArgumentParser(
    description="Compute expected recombinant frequencies and lineage frequencies."
)
parser.add_argument("--metadata",  required=True,
                    help="Test-window metadata TSV.GZ (from get_test_metadata).")
parser.add_argument("--dictionary", required=True,
                    help="Collapsed-lineage CSV produced by collapse_lineages_dict.py.")
parser.add_argument("--out", required=True,
                    help="Output path for pairwise expected-freq table (.csv.gz).")
parser.add_argument("--lineage_freq_out", required=True,
                    help="Output path for single-lineage freq table (.csv.gz).")
args = parser.parse_args()

# ──────────── load & collapse ────────────
print(f"[expected_freq] reading metadata   : {args.metadata}")
df = pd.read_csv(args.metadata, sep="\t", compression="gzip")

print(f"[expected_freq] reading dictionary : {args.dictionary}")
mapping = pd.read_csv(args.dictionary).set_index("variant")["collapsed"].to_dict()

df["collapsed"] = df["Nextclade_pango"].map(mapping)
df = df.dropna(subset=["collapsed"])
if df.empty:
    raise RuntimeError("No sequences remained after collapsing; nothing to compute.")

# ──────────── single-lineage frequencies ────────────
counts = df["collapsed"].value_counts(sort=False)
total  = int(counts.sum())
lineage_df = (
    counts.rename("count")
          .to_frame()
          .assign(total_count=total,
                  p=lambda s: s["count"] / total)
          .reset_index()
          .rename(columns={"index": "Lineage"})
          .sort_values("p", ascending=False)
)
lineage_df.to_csv(args.lineage_freq_out, index=False, compression="gzip",
                  float_format="%.8g")
print(f"[expected_freq] wrote lineage-freq table → {args.lineage_freq_out}")

# ──────────── pairwise expected frequencies ────────────
freqs = counts / total
records = [
    {"Lineage_A": a,
     "Lineage_B": b,
     "pA": freqs[a],
     "pB": freqs[b],
     "Expected_Recombinant_Frequency": 2 * freqs[a] * freqs[b]}
    for a, b in combinations(freqs.index, 2)
]

pairwise_df = (pd.DataFrame(records)
                 .sort_values("Expected_Recombinant_Frequency", ascending=False))
pairwise_df.to_csv(args.out, index=False, compression="gzip", float_format="%.8g")
print(f"[expected_freq] wrote pairwise table      → {args.out}")
