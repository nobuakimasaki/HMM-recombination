#!/usr/bin/env python3
"""
Clean, trim, and attach collapsed lineages to sequence data.
Writes only seq_pango_sorted_{window}.csv.gz.
"""

from collapse_lineage_counts import *
import gzip, argparse, pandas as pd, numpy as np
from Bio import SeqIO

# ---------- CLI ---------- #
parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--trim", type=int, default=0)
parser.add_argument("--collapse_dict", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

# ---------- mapping ---------- #
collapsed_df = pd.read_csv(args.collapse_dict)
collapsed_dict = dict(zip(collapsed_df["variant"], collapsed_df["collapsed"]))

# ---------- helpers ---------- #
def fasta_to_dataframe(fp):
    records, open_func = [], (gzip.open if fp.endswith(".gz") else open)
    with open_func(fp, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            records.append([rec.id, str(rec.seq), rec.description])
    return pd.DataFrame(records, columns=["ID", "Sequence", "Description"])

# ---------- load ---------- #
meta = pd.read_csv(args.metadata, sep="\t", compression="gzip")
meta.columns = meta.columns.str.strip()
seq  = fasta_to_dataframe(args.fasta)

# ---------- merge ---------- #
meta_sel = meta[["strain", "Nextclade_pango", "date"]]
df = (
    meta_sel.merge(seq, left_on="strain", right_on="ID", how="inner")
            .loc[:, ["ID", "Nextclade_pango", "date", "Sequence"]]
)

# ---------- clean ---------- #
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Sequence"] = df["Sequence"].str.replace("[^ACTG]", "N", regex=True)
df["Trimmed"] = df["Sequence"].str[args.trim:-args.trim] if args.trim else df["Sequence"]

# ---------- annotate ---------- #
df["collapsed"] = df["Nextclade_pango"].map(collapsed_dict)

# ---------- save ---------- #
df.sort_values("Nextclade_pango").to_csv(args.out, index=False, compression="gzip")
print(f"[generate_pango_sorted] Saved → {args.out}")
