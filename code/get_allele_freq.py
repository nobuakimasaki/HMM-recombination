### This code is used to obtain the allele frequency matrix from sequences. The sequence data can be generated using aliasing.py.
### The sequence data should have columns 'ID', 'Nextclade_pango', 'collapsed', and 'Trimmed'.

import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter

# Define dictionary between nucleotides and integers
allele_order = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

# Create a parser
parser = argparse.ArgumentParser(description="Generate an allele frequency matrix for Pango lineages.")

# Add arguments
parser.add_argument("--csv", type=str, required=True, help="path to csv file with sequences and corresponding Pango lineages")
parser.add_argument("--out", type=str, required=True, help="output path for CSV file")

# Parse the arguments
args = parser.parse_args()

# Read in sequence file
seq_pango_sorted = pd.read_csv(args.csv, compression='gzip')

# Function to obtain allele proportions within each cluster. Returns list of allele proportions corresponding to each position.
def proportion_allele_counts(sequences):
    # Get sequence length
    sequence_length = len(sequences[0])
    allele_proportions = []

    # For each position
    for pos in range(sequence_length):
        # Get the allele from each sequence
        column = [seq[pos] for seq in sequences]
        # Remove Ns
        filtered_col = [char for char in column if char != 'N']
        # Get allele counts excluding Ns
        counts = Counter(filtered_col)
        total_counts = len(filtered_col)  # Total number of sequences excluding N      

        # If everything was N, assign equal probability to alleles
        if len(filtered_col) == 0:
            counts = Counter(['A', 'T', 'C', 'G'])
            total_counts = 4
            
        # Calculate allele proportions
        proportions = {allele: count / total_counts for allele, count in counts.items()}
        allele_proportions.append(proportions)
    return allele_proportions

print("starting allele frequency calculations")

lineages = {}
for lineage in seq_pango_sorted['collapsed'].unique():
    sequences = seq_pango_sorted[seq_pango_sorted['collapsed'] == lineage]['Trimmed'].tolist()
    print(f"Lineage: {lineage}, Sequence count: {len(sequences)}")
    lineages[lineage] = sequences

# Create a dictionary with the cluster number as the key and a list of sequences within the cluster as the value
lineages = {lineage: seq_pango_sorted[seq_pango_sorted['collapsed'] == lineage]['Trimmed'].tolist() \
              for lineage in seq_pango_sorted['collapsed'].tolist()}

# Calculate and store allele proportions for each cluster (list of allele proportions corresponding to each position)
allele_sums_per_lineage = {}
for lineage, sequences in lineages.items():
    # print(lineage)
    # print(sequences)
    allele_sums_per_lineage[lineage] = proportion_allele_counts(sequences)
    print("calculated allele proportions for one lineage")

# Convert to a wide-format DataFrame with each position as a column
final_df = pd.DataFrame()

# Take each cluster and list of allele proportions
for lineage, allele_sums in allele_sums_per_lineage.items():
    # df has 4 columns, Position, C, A, T, G, indicating the allele proportions for each position
    df = pd.DataFrame(allele_sums).fillna(0).astype(float).reset_index().rename(columns={'index': 'Position'})
    # Convert to long format
    df = df.melt(id_vars='Position', var_name='Allele', value_name='Count')
    # Add cluster label
    df['Lineage'] = lineage
    # Change table so that we have columns for cluster, allele, position, and the entries are the proportions
    df_pivot = df.pivot_table(index=['Lineage', 'Allele'], columns='Position', values='Count', fill_value=0)
    # Concatenate each table (corresponding to one cluster)
    final_df = pd.concat([final_df, df_pivot])

# Resetting the index to have 'Cluster' and 'Allele' as columns
final_df.reset_index(inplace=True)

# Add a column for sorting based on allele order
final_df['Allele_Order'] = final_df['Allele'].map(allele_order)

# Sort the DataFrame by Cluster and then by the custom Allele order
final_df_sorted = final_df.sort_values(by=['Lineage', 'Allele_Order'])

# Drop the auxiliary column used for sorting
final_df_sorted = final_df_sorted.drop(columns=['Allele_Order'])

print("finished calculating allele frequencies")

# Save allele frequency matrix
final_df_sorted.to_csv(args.out, index=False, compression='gzip')

print("saved allele frequency matrix")
