from collapse_lineage_counts import *
import gzip
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from pango_aliasor.aliasor import Aliasor

# Create a parser
parser = argparse.ArgumentParser(description="Get clustered Pango lineages.")

# Add arguments
parser.add_argument("--fasta", type=str, required=True, help="path to fasta file")
parser.add_argument("--metadata", type=str, required=True, help="path to metadata file")
parser.add_argument("--trim", type=int, required=False, help="how many positions to trim from the beginning and end of the genome", default=0)
parser.add_argument("--collapse_threshold", type=int, required=True, help="minimum sequence per collapsed Pango lineage")
parser.add_argument("--out", type=str, required=True, help="output path for CSV file")

# Parse the arguments
args = parser.parse_args()

# Function to read a FASTA file and save contents into a DataFrame
def fasta_to_dataframe(file_path):
    records = []
    # Choose the appropriate open method based on file extension
    open_func = gzip.open if file_path.endswith('.gz') else open

    with open_func(file_path, 'rt') as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            records.append([record.id, str(record.seq), record.description])

    # Create a DataFrame
    df = pd.DataFrame(records, columns=['ID', 'Sequence', 'Description'])
    return df

# Loading in metadata and sequence data
print("loading in aligned.fasta file and metadata file")
meta = pd.read_csv(args.metadata, sep='\t')
seq = fasta_to_dataframe(args.fasta)

# Get the strain and pango information from the metadata and inner join sequences
meta_pango = meta[['strain', 'Nextclade_pango', 'date']]
seq_pango = pd.merge(meta_pango, seq, left_on='strain', right_on='ID', how='inner')
seq_pango = seq_pango[['ID', 'Nextclade_pango', 'date', 'Sequence']]

# Convert the 'date' column to datetime
seq_pango['date'] = pd.to_datetime(seq_pango['date'])

# Sort by pango lineage
seq_pango_sorted = seq_pango.sort_values(by='Nextclade_pango')

# Replace all unknown nucleotides with N
seq_pango_sorted['Sequence'] = seq_pango_sorted['Sequence'].str.replace('[^ACTG]', 'N', regex=True)

# Trim sequence
if args.trim > 0:
    seq_pango_sorted['Trimmed'] = seq_pango_sorted['Sequence'].str[args.trim:-args.trim]
elif args.trim == 0:
    seq_pango_sorted['Trimmed'] = seq_pango_sorted['Sequence']
else:
    raise Exception("Trim length must be a positive integer.")

# Count the number of sequences for each Nextclade_pango lineage
pango_counts = seq_pango_sorted.groupby('Nextclade_pango').size().reset_index(name='Count')

# Rename columns in count file
pango_counts = pango_counts.rename(columns={'Nextclade_pango': 'variant', 'Count': 'sequences'})

# Make a copy of count file
original_pango_counts = pango_counts.copy()

# Collapse Pango lineages using count file
collapse_lineages(pango_counts, args.collapse_threshold, Aliasor(), set())

# Rename columns of collapsed lineage file
pango_counts = pango_counts.rename(columns={'variant': 'collapsed', 'Count': 'sequences'})

# Concatenate the collapsed lineages to count file
pango_dict = pd.concat([original_pango_counts, pango_counts], axis=1)[['variant', 'collapsed']]

# Join collapsed lineages to the sequence file
seq_pango_sorted = pd.merge(seq_pango_sorted, pango_dict, left_on='Nextclade_pango', right_on='variant', how='inner')[['ID', 'Nextclade_pango', 'collapsed', 'date', 'Trimmed']]

# Write sequence file
seq_pango_sorted.to_csv(args.out, index=False, compression='gzip')
