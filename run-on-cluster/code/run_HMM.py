import argparse
import pandas as pd
import numpy as np
from HMM_functions import *
from functools import partial
from scipy.optimize import minimize
import multiprocessing
from Bio import SeqIO
import json
import random
import gzip
import sys
import os

random.seed(4815162342)

allele_order = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

def sample_sequences(sequences, sample_size):
    n = len(sequences)
    if n < sample_size:
        return sequences
    return random.sample(sequences, sample_size) 

def remove_consecutive_duplicates(lst):
    result = [] 
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            result.append([lst[i-1], i, lst[i]])
    if len(result) == 0 and len(lst) > 0:
        result.append([lst[0]])
    return result

# --- new helper ---------------------------
def k_to_s(k, L):
    """Convert expected break-point count k to self-transition prob s."""
    return 1.0 - k / (L - 1)

def load_pi(raw_pi, index_to_pango):
    """Return a numpy vector whose order matches index_to_pango."""
    if raw_pi is None:
        return None                             #  → use uniform later

    # 1) Parse
    if os.path.isfile(raw_pi):
        if raw_pi.endswith(".json"):
            with open(raw_pi) as f:
                pi_dict = json.load(f)
        else:                                   # assume 2-col CSV
            df_pi = pd.read_csv(raw_pi, header=None, names=["collapsed", "prob"])
            pi_dict = dict(zip(df_pi.collapsed, df_pi.prob))
    else:                                       # JSON string
        pi_dict = json.loads(raw_pi)

    # 2) Re-order and fill missing lineages with zero
    vec = np.array([pi_dict.get(lin, 0.0) for lin in index_to_pango], dtype=float)

    # 3) Force to a proper distribution
    if not np.isclose(vec.sum(), 1.0):
        vec = vec / vec.sum()

    return vec

# --- replacement find_parental_clusters ----
def find_parental_clusters(recombinant_numbers, freq, index_to_pango, pi):
    L = len(recombinant_numbers)                       # genome length
    # parameters to optimise: [log_e, k]
    initial_guess = [np.log(0.005), 1.0]                 # e ≈ 0.005,  k ≈ 1
    bounds = [(np.log(1e-8), np.log(0.02)),                     # log_e  ∈ (-18.4 , -3.9]
              (0.0, 3.0)]                              # k      ∈ [0 , 3 ]

    # full model (k free)
    result = minimize(
        nll,
        initial_guess,
        args=(recombinant_numbers, freq, pi),
        bounds=bounds,
        method="L-BFGS-B"
    )

    # null model (k = 0 ⇒ s = 1)
    result_null_model = minimize(
        nll_sigma_1,
        [initial_guess[0]],                            # only log_e
        args=(recombinant_numbers, freq, pi),
        bounds=[bounds[0]],                            # same log_e bound
        method="L-BFGS-B"
    )

    # unpack and back-transform
    est_log_e, est_k = result.x
    est_e = np.exp(est_log_e)
    est_s = k_to_s(est_k, L)

    # downstream decoding
    q_star, delta = viterbi_haplotype_states(recombinant_numbers, freq,
                                             est_s, est_e, pi)
    q_star_group = np.array([index_to_pango[int(num)] for num in q_star])
    mmpp = get_mmpp(recombinant_numbers, freq, est_s, est_e, pi)

    return est_e, est_s, -result.fun, -result_null_model.fun, mmpp, q_star_group

parser = argparse.ArgumentParser(description="Infer parental lineages from recombinant sequences.")
parser.add_argument("--freq", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--optim_out", type=str, required=True)
parser.add_argument("--cpu", type=int, default=5)
parser.add_argument(
    "--pi", type=str, default=None,
    help=(
        "Initial state probabilities.  Either a JSON string "
        'e.g. \'{"BA.1":0.4,"BA.2":0.6}\'  or a path to a JSON/CSV '
        "file with two columns <collapsed,prob>."
    )
)

args = parser.parse_args()

print("Arguments received:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print()

freq = pd.read_csv(args.freq).to_numpy()

# Read and convert test sequences
if args.test.endswith(".csv"):
    test = pd.read_csv(args.test, header=None).to_numpy().tolist()
    ids = [f"ID{i+1}" for i in range(len(test))]
elif args.test.endswith(".fasta") or args.test.endswith(".fasta.gz"):
    if args.test.endswith(".fasta.gz"):
        with gzip.open(args.test, "rt") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
    else:
        with open(args.test, "rt") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
    test = [[allele_order.get(nuc if nuc in "ATCG" else "N", 4) for nuc in str(record.seq).upper()] for record in records]
    ids = [record.id for record in records]
else:
    print("Error: test file must be a CSV or FASTA file.")
    sys.exit(1)

pango_lineages = freq[:,0]
index_to_pango = []
for i in range(len(pango_lineages)):
    if i == 0 or pango_lineages[i] != pango_lineages[i - 1]:
        index_to_pango.append(pango_lineages[i])

pi_vec = load_pi(args.pi, index_to_pango)
find_parental_clusters_partial = partial(
    find_parental_clusters,
    freq=freq,
    index_to_pango=index_to_pango,
    pi=pi_vec,                 # ← new
)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=args.cpu) as pool:
        results = pool.map(find_parental_clusters_partial, test)

    optimization_results = [res[:5] for res in results]
    res_df = pd.DataFrame(optimization_results, columns=["est_e", "est_s", "log_likelihood", "log_likelihood_null", "mmpp"])
    res_df.insert(0, "sequence_id", ids)
    res_df.to_csv(args.optim_out, index=False, compression="gzip")

    inferred_seq = [res[5] for res in results]
    res_list = [remove_consecutive_duplicates(seq) for seq in inferred_seq]
    res_dict = dict(zip(ids, res_list))

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=4)
