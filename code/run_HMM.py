import argparse
import pandas as pd
import numpy as np
from HMM_functions import *
from functools import partial
from scipy.optimize import minimize
import multiprocessing
from Bio import SeqIO
import json

# Define dictionary between nucleotides and integers
allele_order = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

# Function to remove consecutive duplicates (e.g. [a,a,a,b,b,c] becomes [a,b,c])
def remove_consecutive_duplicates(lst):
    result = [] 
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:  # Add to the result if it's not a duplicate of the previous element
            temp = []
            temp.append(lst[i-1])
            temp.append(i)
            temp.append(lst[i])
            result.append(temp)
    if len(result) == 0:
        result.append([lst[0]])
    return result

# Optimize parameters and find the most likely sequence of clusters
def find_parental_clusters(recombinant_numbers, freq, index_to_pango):
    # Initial guess
    initial_guess = [0.1, 0.9] 
    # Perform the minimization
    print("starting optimization")
    result = minimize(nll, initial_guess, args=(recombinant_numbers, freq), 
                      bounds=[(0.00001, 5), (0.001, 1)], method='L-BFGS-B')
    result_null_model = minimize(nll_sigma_1, initial_guess[0], args=(recombinant_numbers, freq), 
                      bounds=[(0.00001, 5)], method='L-BFGS-B')
    print("optimal solution (e and s): {}".format(result.x))
    est_e = result.x[0]
    est_s = result.x[1]
    print("log-likelihood: {}".format(-result.fun))
    # Viterbi algorithm
    print("starting Viterbi")
    q_star, delta = viterbi_haplotype_states(recombinant_numbers, freq, est_s, est_e)
    q_star_group = np.array([index_to_pango[int(num)] for num in q_star])

    return est_e, est_s, -result.fun, -result_null_model.fun, q_star_group

# Create a parser
parser = argparse.ArgumentParser(description="Generate an allele frequency matrix for Pango lineages.")

# Add arguments
parser.add_argument("--freq", type=str, required=True, help="path to allele frequency matrix")
parser.add_argument("--test", type=str, required=True, help="path to list of sequences we want to infer parental lineages")
parser.add_argument("--out", type=str, required=True, help="output path (json)")
parser.add_argument("--optim_out", type=str, required=True, help="output path for optimization results")
# parser.add_argument("--plot", type=bool, default=False, help="specify whether to plot the inferred hidden states")
# parser.add_argument("--plot_out", type=str, default="../output/inferred/inferred_lineages.png", help="path for saving plot")

# Parse the arguments
args = parser.parse_args()

# Read allele frequencies and test set
freq = pd.read_csv(args.freq).to_numpy()

if args.test.endswith(".csv"):
    test = pd.read_csv(args.test, header=None).to_numpy().tolist()
elif args.test.endswith(".fasta"):
    test = [[allele_order.get(nuc if nuc in "ATCG" else "N", 4) for nuc in str(record.seq).upper()] for record in SeqIO.parse(args.test, "fasta")]
else:
    print("Error: test file must be a FASTA file.")
    sys.exit() 

# Obtain the list of Pango lineages in the first column of the allele frequency matrix (in the order in which they appear)
pango_lineages = freq[:,0]
index_to_pango = []
for i in range(len(pango_lineages)):
    if i == 0 or pango_lineages[i] != pango_lineages[i - 1]:
        index_to_pango.append(pango_lineages[i])

# Define new function with cluster_array fixed for multiprocessing
find_parental_clusters_partial = partial(find_parental_clusters, freq=freq, index_to_pango=index_to_pango)

# Use Pool to parallelize the process
if __name__ == '__main__':
    print(f"Analyzing {len(test)} Sequences.")
    # Get cpu count
    cpu_count = multiprocessing.cpu_count()-1
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Map the combined list to the worker processes
        results = pool.map(find_parental_clusters_partial, test)

    # Extract only the first four elements from each result
    optimization_results = [res[:4] for res in results]
    # Create DataFrame
    res_df = pd.DataFrame(optimization_results, columns=["est_e", "est_s", "log_likelihood", "log_likelihood_null"])
    # Save to a CSV file
    res_df.to_csv(args.optim_out, index=False)

    # Get inferred sequences of hidden states
    inferred_seq = [res[4] for res in results]

    # Get a representation of each sequence
    res_list = []
    for i in range(len(inferred_seq)):
        temp = remove_consecutive_duplicates(inferred_seq[i])
        res_list.append(temp)

    res_dict = {f"ID{i+1}": v for i, v in enumerate(res_list)}

    # Save to JSON file
    with open(args.out, "w") as f:
        json.dump(res_dict, f, indent=4)

    # if args.plot:

    #     print("plotting inferred parental lineages")

    #     import seaborn as sns
    #     import matplotlib.patches as mpatches
    #     import matplotlib.pyplot as plt

    #     # Get unique lineage labels and map them to numeric values
    #     unique_lineages = sorted(set(lin for seq in inferred_seq for lin in seq))
    #     lineage_map = {lin: i for i, lin in enumerate(unique_lineages)}

    #     # Convert sequences to numeric matrices
    #     lineage_plot_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in inferred_seq])

    #     # Choose a colormap (same for both sets)
    #     num_colors = len(unique_lineages)
    #     cmap = sns.color_palette("tab10", num_colors)
    #     cmap = plt.cm.colors.ListedColormap(cmap)

    #     # Create a figure with a single subplot
    #     fig, ax = plt.subplots(figsize=(10, 10)) 

    #     # Lineage sequences visualization
    #     ax.imshow(lineage_plot_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    #     ax.set_xlabel("Position")
    #     ax.set_ylabel("Sequence Index")
    #     ax.set_title("Inferred Lineages")

    #     # Legend
    #     legend_patches = [mpatches.Patch(color=cmap(i), label=lineage) for lineage, i in lineage_map.items()]
    #     fig.legend(handles=legend_patches, title="Lineage", bbox_to_anchor=(1.05, 0.5), loc='center left')

    #     # Show the plot
    #     plt.tight_layout()

    #     # Save figure as PNG
    #     plt.savefig(args.plot_out, dpi=100, bbox_inches='tight')


