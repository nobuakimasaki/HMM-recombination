### This code is used to simulate recombinant sequences. The sequence data can be generated using aliasing.py.
### The sequence data should have columns 'ID', 'Nextclade_pango', 'collapsed', and 'Trimmed'.

import argparse
import pandas as pd
import numpy as np
import random
from Bio import SeqIO
from pathlib import Path

# Define dictionary between nucleotides and integers
allele_order = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

# Create a parser
parser = argparse.ArgumentParser(description="Generate recombinant sequences based on an aligned fasta file.")

# Add arguments
parser.add_argument("--csv", type=str, required=True, help="path to csv file with sequences and corresponding Pango lineages")
parser.add_argument("--rate", type=float, default=0.0002, help="mutation rate")
parser.add_argument("--n_single", type=int, default=800, help="number of recombinant sequences to generate (single breakpoint)")
parser.add_argument("--n_double", type=int, default=200, help="number of recombinant sequences to generate (two breakpoints)")
parser.add_argument("--n_control", type=int, default=1000, help="number of control sequences to generate")
parser.add_argument("--plot", action="store_true", help="enable plotting of the simulated sequences")
parser.add_argument("--no-plot", action="store_false", dest="plot", help="disable plotting of the simulated sequences")
parser.set_defaults(plot=False)

# Parse the arguments
args = parser.parse_args()

# Read in sequence file
seq_pango_sorted = pd.read_csv(args.csv)
n_rows = len(seq_pango_sorted)      
print(f"Rows: {n_rows}")

# Read in empirical mutation counts
def load_empirical_counts(path="output/mutation_counts/mutation_count"):
    with Path(path).open() as f:
        return [int(line.strip()) for line in f if line.strip()]

emp_counts = load_empirical_counts("output/mutation_counts/mutation_count")
print(emp_counts)

random.seed(26)
np.random.seed(26)

### Function to calculate Hamming distance between sequences ignoring 'N's
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2) if c1 != 'N' and c2 != 'N')

# Function to introduce mutations in a numeric DNA sequence
# def mutate_numeric_sequence(sequence, mutation_rate=0.0002):
#     # Copy sequence
#     mutated_sequence = sequence.copy()
#     # For each position of the sequence
#     for i in range(len(mutated_sequence)):
#         # Mutate the position with a certain probability
#         if random.random() < mutation_rate:
#             possible_mutations = [0, 1, 2, 3, 4]
#             possible_mutations.remove(mutated_sequence[i])  # Exclude the current nucleotide
#             mutated_sequence[i] = random.choice(possible_mutations)  # Mutate to a different nucleotide
#     return mutated_sequence

# 2) Mutate using an empirical draw of the total number of mutations -----------
def mutate_numeric_sequence_empirical(sequence, emp_counts):
    """
    Mutate exactly m sites, where m is drawn from the empirical counts.
    Uses Python's global RNG `random` (seeded by random.seed(...)).
    """
    if not emp_counts:
        raise ValueError("emp_counts is empty.")
    m = int(random.choice(emp_counts))           # draw total mutation count
    m = min(m, len(sequence))                    # cap at sequence length
    if m == 0:
        return sequence.copy()

    idxs = random.sample(range(len(sequence)), m)  # choose m distinct indices

    mutated = sequence.copy()
    for i in idxs:
        current = mutated[i]
        choices = [0, 1, 2, 3, 4]
        if current in choices:
            choices.remove(current)
        mutated[i] = random.choice(choices)
    return mutated

# Function to create a recombinant sequence from two sequences, seq1 and seq2 should be strings
def create_recombinant(seq1, seq2):
    # If the two sequences are the same, return False
    if seq1 == seq2:
        return False, False
    # If not,
    else:
        # Choose a random crossover point
        crossover_point = random.randint(1, len(seq1) - 1)
        # Combine the first part of seq1 with the second part of seq2
        recombinant_sequence = seq1[:crossover_point] + seq2[crossover_point:]
        # Return False if the generated recombinant sequence is the same as one of the two sequences
        if hamming_distance(recombinant_sequence, seq1) <= 1 or hamming_distance(recombinant_sequence, seq2) <= 1:
            return False, False

    return ''.join(recombinant_sequence), crossover_point

# Function to create a recombinant sequence from two sequences (two breakpoints). seq1 and seq2 should be strings
def create_recombinant2(seq1, seq2, all_pairs):
    # If the two sequences are the same, return False
    if seq1 == seq2:
        return False, False, False
    # If not,
    else:
        # Sample from all possible pairs of crossover points
        crossover_point1, crossover_point2 = random.choice(all_pairs)
        # Combine seq1 and seq2
        recombinant_sequence = seq1[:crossover_point1] + seq2[crossover_point1:crossover_point2] + seq1[crossover_point2:]
        # Return False if the generated recombinant sequence is the same as one of the two sequences
        if hamming_distance(recombinant_sequence, seq1) <= 1 or hamming_distance(recombinant_sequence, seq2) <= 1:
            return False, False, False

    return ''.join(recombinant_sequence), crossover_point1, crossover_point2

### Sequence generation starts here
print("starting sequence generation")

# Lists to store results
recombinant_numbers_list = []
sampled_sequences = []
crossover_points = []

print("generating sequences with one breakpoint")
# Generate 800 recombinant sequences with one breakpoint
while len(recombinant_numbers_list) < args.n_single:
    # Randomly choose two different sequences
    sampled_df = seq_pango_sorted.sample(n=2)

    # If Pango lineage is the same, skip this iteration
    if sampled_df.iloc[0]['collapsed'] == sampled_df.iloc[1]['collapsed']:
        continue
    
    # Retrieve the sequences based on the chosen indices
    seq1 = sampled_df.iloc[0]['Trimmed']
    seq2 = sampled_df.iloc[1]['Trimmed']

    # Create a recombinant sequence and store the crossover point
    recombinant_sequence, cross_point = create_recombinant(seq1, seq2)

    # If create_recombinant successfully created a recombinant sequence
    if recombinant_sequence != False:

        # Convert the sequences to a list of numbers based on allele_order
        seq1_numbers = [allele_order[allele] for allele in seq1]
        seq2_numbers = [allele_order[allele] for allele in seq2]
        recombinant_numbers = [allele_order[allele] for allele in recombinant_sequence]
        
        # Introduce additional mutations into the recombinant sequence
        final_recombinant_numbers = mutate_numeric_sequence_empirical(recombinant_numbers, emp_counts=emp_counts)
        
        # Store the results
        recombinant_numbers_list.append(final_recombinant_numbers)
        sampled_sequences.append([sampled_df.iloc[0]['ID'], sampled_df.iloc[1]['ID']])
        crossover_points.append([cross_point])
        
# Define all possible pairs of crossover points
all_pairs = [(i, j) for i in range(1, len(seq1) - 2) for j in range(i + 1, len(seq1) - 1)]

print("generating sequences with two breakpoints")
# Generate 200 recombinant sequences with two breakpoints
while len(recombinant_numbers_list) < args.n_single + args.n_double:
    # Randomly choose two different sequences
    sampled_df = seq_pango_sorted.sample(n=2)

    # If Pango lineage is the same, skip this iteration
    if sampled_df.iloc[0]['collapsed'] == sampled_df.iloc[1]['collapsed']:
        continue
    
    # Retrieve the sequences based on the chosen indices
    seq1 = sampled_df.iloc[0]['Trimmed']
    seq2 = sampled_df.iloc[1]['Trimmed']

    # Create a recombinant sequence and store the crossover point
    recombinant_sequence, cross_point1, cross_point2 = create_recombinant2(seq1, seq2, all_pairs)

    # If create_recombinant successfully created a recombinant sequence
    if recombinant_sequence != False:

        # Convert the sequences to a list of numbers based on allele_order
        seq1_numbers = [allele_order[allele] for allele in seq1]
        seq2_numbers = [allele_order[allele] for allele in seq2]
        recombinant_numbers = [allele_order[allele] for allele in recombinant_sequence]
        
        # Introduce additional mutations into the recombinant sequence
        final_recombinant_numbers = mutate_numeric_sequence_empirical(recombinant_numbers, emp_counts=emp_counts)
        
        # Store the results
        recombinant_numbers_list.append(final_recombinant_numbers)
        sampled_sequences.append([sampled_df.iloc[0]['ID'], sampled_df.iloc[1]['ID']])
        crossover_points.append([cross_point1, cross_point2])

print("finished generating recombinant sequences")

# Lists to store results
control_numbers_list = []
sampled_sequences_control = []

print("generating control sequences")
# Generate 1000 non-recombinant sequences
while len(control_numbers_list) < args.n_control:
    # Randomly choose one sequence
    sampled_df = seq_pango_sorted.sample(n=1)
    
    # Retrieve the sequence
    seq1 = sampled_df.iloc[0]['Trimmed']

    # Convert the sequence to a list of numbers based on allele_order
    seq1_numbers = [allele_order[allele] for allele in seq1]
    
    # Introduce additional mutations into the sequence
    final_nonrecombinant_numbers = mutate_numeric_sequence_empirical(seq1_numbers, emp_counts=emp_counts)
    
    # Store the results
    control_numbers_list.append(final_nonrecombinant_numbers)
    sampled_sequences_control.append([sampled_df.iloc[0]['ID']])

print("finished generating non-recombinant sequences (control set)")

recombinant_numbers_list_df = pd.DataFrame(recombinant_numbers_list)
recombinant_numbers_list_df.to_csv("output/simulated_sequences/recombinants.csv", index=False, header=False)

sampled_sequences_df = pd.DataFrame(sampled_sequences)
sampled_sequences_df.to_csv("output/simulated_sequences/sampled_sequences.csv", index=False, header=False)

crossover_points_df = pd.DataFrame(crossover_points)
crossover_points_df.to_csv("output/simulated_sequences/breakpoints.csv", index=False, header=False)

control_numbers_list_df = pd.DataFrame(control_numbers_list)
control_numbers_list_df.to_csv("output/simulated_sequences/controls.csv", index=False, header=False)

sampled_sequences_control_df = pd.DataFrame(sampled_sequences_control)
sampled_sequences_control_df.to_csv("output/simulated_sequences/sampled_sequences_control.csv", index=False, header=False)

print("saved simulated sequences")

if args.plot and len(sampled_sequences) > 30:
    print("recombinant sequence count exceeds 30 so omitting plotting")

if args.plot and len(sampled_sequences) <= 30:
    print("plotting parental lineages for simulated sequences")

    import seaborn as sns
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Function to generate the true sequence of Pango lineages for each recombinant
    # tuples: a list of the two Pango lineages used to create each recombinant sequence
    # positions: breakpoint positions (1 or 2)
    # N: length of entire sequence
    def create_recombinant_sequences(tuples, positions, N):
        sequences = []
        for (first, second), position in zip(tuples, positions):
            if len(position) == 1:
                sequence = [first] * position[0] + [second] * (N - position[0])
                sequences.append(sequence)
            else:
                pos1 = position[0]
                pos2 = position[1]
                sequence = [first] * pos1 + [second] * (pos2 - pos1) + [first] * (N - pos2)
                sequences.append(sequence)
        return sequences

    # We save the true Pango lineages leading to each recombinant here:
    true_pango_set_list = []
    # Filter the reference set for each tuple of chosen IDs
    for id_pair in sampled_sequences:
        true_pango_set = []
        # Filter rows matching either ID in the tuple, and get the 'Nextclade_pango' column
        true_pango_set.extend(seq_pango_sorted[seq_pango_sorted['ID'] == id_pair[0]]['collapsed'].tolist())
        true_pango_set.extend(seq_pango_sorted[seq_pango_sorted['ID'] == id_pair[1]]['collapsed'].tolist())
        true_pango_set_list.append(true_pango_set)

    # Generate true recombinant sequences
    recombinant_sequences = create_recombinant_sequences(true_pango_set_list, crossover_points, len(recombinant_numbers_list[0]))

    # Get unique lineage labels and map them to numeric values
    unique_lineages = sorted(set(lin for seq in recombinant_sequences for lin in seq))
    lineage_map = {lin: i for i, lin in enumerate(unique_lineages)}

    # Convert sequences to numeric matrices
    lineage_plot_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in recombinant_sequences])

    # Choose a colormap (same for both sets)
    num_colors = len(unique_lineages)
    cmap = sns.color_palette("tab10", num_colors)
    cmap = plt.cm.colors.ListedColormap(cmap)

    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 10)) 

    # Lineage sequences visualization
    ax.imshow(lineage_plot_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xlabel("Position")
    ax.set_ylabel("Sequence Index")
    ax.set_title("True Lineages")

    # Legend
    legend_patches = [mpatches.Patch(color=cmap(i), label=lineage) for lineage, i in lineage_map.items()]
    fig.legend(handles=legend_patches, title="Lineage", bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Show the plot
    plt.tight_layout()

    # Save figure as PNG
    plt.savefig('output/simulated_sequences/recombinants.png', dpi=100, bbox_inches='tight')
