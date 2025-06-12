from collapse_lineage_counts import collapse_lineages
import pandas as pd
import numpy as np
from pango_aliasor.aliasor import Aliasor
import argparse
import glob
import os

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Collapse Pango lineages using one large metadata file.")
parser.add_argument("--reference_metadata", type=str, required=True,
                    help="Path to the large metadata file (e.g. aligned.min-date....tsv.gz)")
parser.add_argument("--metadata_glob", type=str, required=True,
                    help="Glob pattern for test metadata files to update, e.g., output/sliding_windows/test_*.metadata.tsv")
parser.add_argument("--collapse_threshold", type=int, required=True,
                    help="Minimum number of sequences per collapsed lineage.")
parser.add_argument("--out", type=str, required=True,
                    help="Output path for collapsed lineage dictionary CSV.")

args = parser.parse_args()

# --- Load large reference metadata file ---
print(f"Reading large metadata file: {args.reference_metadata}")
ref_meta = pd.read_csv(args.reference_metadata, sep='\t', compression='gzip')
if 'Nextclade_pango' not in ref_meta.columns:
    raise ValueError(f"'Nextclade_pango' column not found in reference metadata.")

# --- Count and collapse lineages ---
pango_counts = ref_meta.groupby('Nextclade_pango').size().reset_index(name='sequences')
pango_counts = pango_counts.rename(columns={'Nextclade_pango': 'variant'})
pango_counts['original_variant'] = pango_counts['variant']

# --- Collapse (modifies 'variant' column in-place)
collapse_lineages(pango_counts, args.collapse_threshold, Aliasor(), set())

# --- Generate collapsed dictionary
collapsed_df = pango_counts[['original_variant', 'variant']].rename(
    columns={'original_variant': 'variant', 'variant': 'collapsed'}
)
collapsed_dict = dict(zip(collapsed_df['variant'], collapsed_df['collapsed']))

# --- Save collapsed mapping
collapsed_df.to_csv(args.out, index=False)
print(f"Collapsed mapping saved to {args.out}")

# --- Apply mapping to test metadata files ---
test_files = sorted(glob.glob(args.metadata_glob))
print(f"Found {len(test_files)} test metadata files.")

for file in test_files:
    df = pd.read_csv(file, sep='\t')
    if 'Nextclade_pango' not in df.columns:
        raise ValueError(f"'Nextclade_pango' column not found in {file}")
    df['collapsed'] = df['Nextclade_pango'].map(collapsed_dict)
    df['collapsed'] = df['collapsed'].fillna('other')  # Optional: handle unmapped
    df.to_csv(file, sep='\t', index=False)
    print(f"Updated {file} with collapsed lineages.")

print("\nDone.")
