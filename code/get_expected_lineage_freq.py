import pandas as pd
import argparse
import os
from itertools import combinations

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Compute expected recombinant frequencies from metadata.")
parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file (.tsv.gz with 'collapsed' column)")
parser.add_argument("--out", type=str, required=True, help="Output path for expected recombinant frequencies (.csv.gz)")

args = parser.parse_args()

# --- Load metadata ---
print(f"Loading metadata from {args.metadata}")
df = pd.read_csv(args.metadata, sep='\t', compression='gzip')
if 'collapsed' not in df.columns:
    raise ValueError("Missing 'collapsed' column in metadata file.")

# --- Drop missing collapsed values and count frequencies ---
collapsed_counts = df['collapsed'].dropna().value_counts()
total = collapsed_counts.sum()
collapsed_freqs = collapsed_counts / total

# --- Compute expected recombinant frequencies ---
recombinants = []

for (a, b) in combinations(collapsed_freqs.index, 2):  # A ≠ B
    pA = collapsed_freqs[a]
    pB = collapsed_freqs[b]
    freq = 2 * pA * pB
    recombinants.append({
        "Lineage_A": a,
        "Lineage_B": b,
        "pA": round(pA, 6),
        "pB": round(pB, 6),
        "Expected_Recombinant_Frequency": round(freq, 6)
    })

# --- Save to CSV ---
recombinant_df = pd.DataFrame(recombinants)
recombinant_df = recombinant_df.sort_values(by="Expected_Recombinant_Frequency", ascending=False)

recombinant_df.to_csv(args.out, index=False, compression="gzip")

print(f"Saved expected recombinant frequencies to: {args.out}")
