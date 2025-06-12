from collapse_lineage_counts import *
import gzip
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Clean, trim, and attach collapsed lineages to sequence data.")

parser.add_argument("--fasta", type=str, required=True, help="Path to aligned FASTA file (.fasta or .fasta.gz)")
parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file (.tsv)")
parser.add_argument("--trim", type=int, default=0, help="Positions to trim from both ends of each sequence")
parser.add_argument("--collapse_dict", type=str, required=True, help="Path to collapsed lineage CSV (with 'variant' and 'collapsed')")
parser.add_argument("--out", type=str, required=True, help="Output path for CSV file (.csv.gz)")

args = parser.parse_args()

# --- Load collapsed mapping ---
collapsed_df = pd.read_csv(args.collapse_dict)
collapsed_dict = dict(zip(collapsed_df['variant'], collapsed_df['collapsed']))

# --- Read FASTA into DataFrame ---
def fasta_to_dataframe(file_path):
    records = []
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, 'rt') as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            records.append([record.id, str(record.seq), record.description])
    return pd.DataFrame(records, columns=['ID', 'Sequence', 'Description'])

print("Loading FASTA and metadata...")
meta = pd.read_csv(args.metadata, sep='\t', compression='gzip')
meta.columns = meta.columns.str.strip()  # remove accidental whitespace/newlines
seq = fasta_to_dataframe(args.fasta)

print("Metadata columns:", meta.columns.tolist())

# --- Join metadata and sequence ---
meta_pango = meta[['strain', 'Nextclade_pango', 'date']]
seq_pango = pd.merge(meta_pango, seq, left_on='strain', right_on='ID', how='inner')
seq_pango = seq_pango[['ID', 'Nextclade_pango', 'date', 'Sequence']]

# --- Clean and trim ---
seq_pango['date'] = pd.to_datetime(seq_pango['date'], errors='coerce')
seq_pango['Sequence'] = seq_pango['Sequence'].str.replace('[^ACTG]', 'N', regex=True)
if args.trim > 0:
    seq_pango['Trimmed'] = seq_pango['Sequence'].str[args.trim:-args.trim]
else:
    seq_pango['Trimmed'] = seq_pango['Sequence']

# --- Attach collapsed lineage ---
seq_pango['collapsed'] = seq_pango['Nextclade_pango'].map(collapsed_dict)

# --- Sort and write output ---
seq_pango_sorted = seq_pango.sort_values(by='Nextclade_pango')
seq_pango_sorted.to_csv(args.out, index=False, compression='gzip')

print(f"Done. Output saved to {args.out}")

# --- Write summary file with number of unique collapsed lineages ---
summary_path = args.out.replace(".csv.gz", ".collapsed_lineage_count.txt")
n_unique = seq_pango_sorted['collapsed'].nunique()

with open(summary_path, 'w') as f:
    f.write(f"{n_unique}\n")

print(f"Wrote {n_unique} unique collapsed lineages to: {summary_path}")