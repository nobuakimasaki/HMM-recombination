#!/usr/bin/env python3
"""
collapse_lineages_dict.py

Read a Zstandard-compressed reference metadata file,
collapse low-frequency Pango lineages into higher-level aliases,
and write a two-column CSV mapping original → collapsed lineage.

Example
-------
python3 collapse_lineages_dict.py \
        --reference_metadata output/aligned.min-date.2020-09-01.max-date.2024-03-31.division.England.metadata.tsv.zst \
        --collapse_threshold 10000 \
        --out output/collapse_dict.csv
"""
from collapse_lineage_counts import collapse_lineages
from pango_aliasor.aliasor import Aliasor
import pandas as pd
import argparse

# ------------------------- argument parsing ------------------------- #
parser = argparse.ArgumentParser(
    description="Collapse Pango lineages in a single large metadata file."
)
parser.add_argument(
    "--reference_metadata",
    required=True,
    help="Zstd-compressed metadata TSV containing a 'Nextclade_pango' column.",
)
parser.add_argument(
    "--collapse_threshold",
    type=int,
    required=True,
    help="Minimum number of sequences per collapsed lineage.",
)
parser.add_argument(
    "--out",
    required=True,
    help="Path to write the collapsed-lineage dictionary CSV.",
)
args = parser.parse_args()

# -------------------------- load metadata --------------------------- #
print(f"[collapse_dict] Reading {args.reference_metadata}")
ref_meta = pd.read_csv(args.reference_metadata, sep="\t", compression="zstd")
if "Nextclade_pango" not in ref_meta.columns:
    raise ValueError("'Nextclade_pango' column not found in reference metadata.")

# ---------------------- collapse lineages --------------------------- #
counts = (
    ref_meta.groupby("Nextclade_pango")
    .size()
    .reset_index(name="sequences")
    .rename(columns={"Nextclade_pango": "variant"})
)
counts["original_variant"] = counts["variant"]

collapse_lineages(counts, args.collapse_threshold, Aliasor(), set())

collapsed_df = counts[["original_variant", "variant"]].rename(
    columns={"original_variant": "variant", "variant": "collapsed"}
)
collapsed_df.to_csv(args.out, index=False)
print(f"[collapse_dict] Collapsed mapping saved → {args.out}")
