import io, ast             
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.stats.proportion import proportion_confint
from Bio import SeqIO
import zstandard as zstd  
from pathlib import Path
from itertools import starmap

### Function to generate the true sequence of Pango lineages for each recombinant
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

### Function to calculate Hamming distance between sequences ignoring 'N's
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2) if c1 != 'N' and c2 != 'N')

# Set seed for reproducibility
np.random.seed(108)

# Load results
optim_res = pd.read_csv("../sim-3/output/inferred/optimization_results.csv", compression="gzip")
optim_res_controls = pd.read_csv("../sim-3/output/inferred/optimization_results_controls.csv", compression="gzip")

with gzip.open("../sim-3/output/inferred/inferred_lineages.json", "rt") as f:   # “rt” = read-text
    res = json.load(f)
with gzip.open("../sim-3/output/inferred/inferred_lineages_controls.json", "rt") as f:
    res_controls = json.load(f)

# Load reference set
ref = pd.read_csv("../run-on-cluster-3/real-data-analysis/output/sliding_windows/seq_pango_sorted/seq_pango_sorted_2022-11-06_2022-12-11.csv.gz")

# Load sampled IDs and breakpoints
sampled_seq = pd.read_csv("../sim-3/output/simulated_sequences/sampled_sequences.csv", header = None).values.tolist()
sampled_seq_controls = pd.read_csv("../sim-3/output/simulated_sequences/sampled_sequences_control.csv", header = None).values.tolist()
breakpoints = pd.read_csv("../sim-3/output/simulated_sequences/breakpoints.csv", header = None).values.tolist()
# Remove 'nan' values from sublists
breakpoints = [[int(x) for x in sublist if not (isinstance(x, float) and np.isnan(x))] for sublist in breakpoints]

""" SECTION: Obtaining Parameters """
n_cases = len(res)
n_controls = len(res_controls)
N = len(ref['Trimmed'][0])
Pango_lineage_set = sorted(set(ref['collapsed']))
M = len(Pango_lineage_set)

print(f"Number of cases: {n_cases}")
print(f"Number of controls: {n_controls}")
print(f"Length of trimmed genome: {N}")
print(f"Number of Pango lineages after collapsing: {M}")
print(f"Pango lineages after collapsing: {Pango_lineage_set}")

""" SECTION: Calculate Sensitivity and Specificity """
print()
print("SECTION: Calculate Sensitivity and Specificity")
cases_positive_index_breakpnt = np.empty(n_cases)
controls_positive_index_breakpnt = np.empty(n_controls)

# Obtain vector of 0s and 1s based on whether we predicted the sequence as a recombinant
for i, breakpnt in enumerate(res.values()):
	# If the first entry is not length 1 (meaning there is a breakpoint)
	if len(breakpnt[0]) != 1:
		cases_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		cases_positive_index_breakpnt[i] = 0

k_sens = sum(cases_positive_index_breakpnt)
sens_breakpnt = k_sens/n_cases

# Obtain vector of 0s and 1s based on whether we predicted the sequence as a recombinant
for i, breakpnt in enumerate(res_controls.values()):
	# If the first entry is not length 1 (meaning there is a breakpoint)
	if len(breakpnt[0]) != 1:
		controls_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		controls_positive_index_breakpnt[i] = 0

k_spec = sum(controls_positive_index_breakpnt == 0)
spec_breakpnt = k_spec/n_controls

ci_sens = proportion_confint(k_sens, n_cases,    alpha=0.05, method="beta")
ci_spec = proportion_confint(k_spec, n_controls, alpha=0.05, method="beta")

print(f"Sensitivity (breakpoint): {sens_breakpnt:.3f} "
      f"[95 % CI {ci_sens[0]:.3f}, {ci_sens[1]:.3f}]")
print(f"Specificity (breakpoint): {spec_breakpnt:.3f} "
      f"[95 % CI {ci_spec[0]:.3f}, {ci_spec[1]:.3f}]")

print(sum(cases_positive_index_breakpnt))
print(sum(controls_positive_index_breakpnt == 0))

""" SECTION: Calculate Predicted Proportion of Recombinants """  
print()
print("SECTION: Calculate Predicted Proportion of Recombinants")

# # Bootstrap the vector of 0s and 1s to get a bootstrap percentile interval for sensitivity and specificity
# boot_cases_breakpnt_list = []
# boot_controls_breakpnt_list = []

# for _ in range(500):
#     boot_cases_breakpnt = np.random.choice(cases_positive_index_breakpnt, size=n_cases, replace=True)
#     boot_controls_breakpnt = np.random.choice(controls_positive_index_breakpnt, size=n_controls, replace=True)
#     boot_cases_breakpnt_list.append(boot_cases_breakpnt)
#     boot_controls_breakpnt_list.append(boot_controls_breakpnt)

# boot_sens_breakpnt = [sum(positive_index)/n_cases for positive_index in boot_cases_breakpnt_list]
# boot_spec_breakpnt = [sum(positive_index == 0)/n_controls for positive_index in boot_controls_breakpnt_list]

# # Predicted proportion of recombinants
# prop_breakpnt = (sum(cases_positive_index_breakpnt) + sum(controls_positive_index_breakpnt))/(n_cases + n_controls)

# boot_prop_breakpnt = []
# for i in range(len(boot_cases_breakpnt_list)):
# 	boot_prop_breakpnt.append((sum(boot_cases_breakpnt_list[i]) + sum(boot_controls_breakpnt_list[i]))/(n_cases + n_controls))

# # Calculate bootstrap percentile intervals
# ci_lower_prop_breakpnt, ci_upper_prop_breakpnt = np.percentile(boot_prop_breakpnt, [2.5, 97.5])

# print(f"Predicted proportion of recombinants using breakpoint: {prop_breakpnt:.3f} (95% CI: [{ci_lower_prop_breakpnt:.3f}, {ci_upper_prop_breakpnt:.3f}])")
# # print(f"Predicted proportion of recombinants using s: {prop_s:.3f} (95% CI: [{ci_lower_prop_s:.3f}, {ci_upper_prop_s:.3f}])")

""" SECTION: Calculate Predicted Proportion of Recombinants """
print()
print("SECTION: Calculate Predicted Proportion of Recombinants (Šidák CI)")

# Selected hypothetical true prevalences (theta)
theta_selected = np.array([0.000, 0.001, 0.005, 0.010, 0.015, 0.020])
theta_selected = np.round(theta_selected, 3)

# Šidák-adjusted marginal alpha for 95% joint coverage over (Se, Sp)
alpha_joint = 0.05
alpha_sidak = 1 - np.sqrt(1 - alpha_joint)     # ≈ 0.025317...

# Recompute Se/Sp marginal CIs at Šidák level (Clopper–Pearson/"beta")
se_L, se_U = proportion_confint(int(k_sens), n_cases,    alpha=alpha_sidak, method="beta")
sp_L, sp_U = proportion_confint(int(k_spec), n_controls, alpha=alpha_sidak, method="beta")

# Point estimates from above
Se_hat = sens_breakpnt
Sp_hat = spec_breakpnt

def fmt_sig2(x: float) -> str:
    """Format with two significant digits (no forced trailing zeros)."""
    return f"{x:.2g}"

print(f"Using Šidák-adjusted marginals (alpha' = {alpha_sidak:.6f}); target joint coverage = 95%.")
print("theta    p_hat    CI_lower   CI_upper")

for theta in theta_selected:
    # Point estimate of predicted-positive rate for this theta
    p_hat = theta * Se_hat + (1 - theta) * (1 - Sp_hat)

    # Endpoint-mapped CI using monotonicity in Se (↑) and Sp (↓)
    p_lo = theta * se_L + (1 - theta) * (1 - sp_U)
    p_hi = theta * se_U + (1 - theta) * (1 - sp_L)

    # Clip numerically to [0,1]
    p_hat = float(np.clip(p_hat, 0.0, 1.0))
    p_lo  = float(np.clip(p_lo,  0.0, 1.0))
    p_hi  = float(np.clip(p_hi,  0.0, 1.0))

    print(f"{theta:0.3f}   {fmt_sig2(p_hat):>6}   {fmt_sig2(p_lo):>6}   {fmt_sig2(p_hi):>6}")

""" SECTION: Calculate Position-By-Position Accuracy """
print()
print("SECTION: Calculate Position-By-Position Accuracy")

# We save the true Pango lineages leading to each recombinant here:
true_pango_set_list = []
# Filter the reference set for each tuple of chosen IDs
for id_pair in sampled_seq:
    true_pango_set = []
    # Filter rows matching either ID in the tuple, and get the collapsed Pango lineage
    true_pango_set.extend(ref[ref['ID'] == id_pair[0]]['collapsed'].tolist())
    true_pango_set.extend(ref[ref['ID'] == id_pair[1]]['collapsed'].tolist())
    true_pango_set_list.append(true_pango_set)
# Generate true recombinant sequences
true_recombinant_sequences = np.array(create_recombinant_sequences(true_pango_set_list, breakpoints, N))
true_recombinant_sequences_flattened = true_recombinant_sequences.flatten()

# Get inferred breakpoints and the lineages to the left and right
inferred_breakpoints_and_lineages = [res_breakpnt for res_breakpnt in res.values()]
# Generate inferred recombinant sequences
inferred_recombinant_sequences = []
for sublist in inferred_breakpoints_and_lineages:
	# sublist is a list of lists containing breakpoints and inferred lineages (e.g. [A, 200, B])
	if len(sublist[0]) == 1: # If only one lineage is inferred
		inferred_recombinant_sequences.append(sublist[0]*N)
	else: # If more than one lineage is inferred
		temp = [sublist[0][0]]*N # First create a sequence with just the first inferred lineage
		for subsublist in sublist: # For each new breakpoint
			temp[subsublist[1]:N] = [subsublist[2]] * (N - subsublist[1]) # The sequence after the breakpoint should be a new lineage
		inferred_recombinant_sequences.append(temp)
# Inferred recombinant sequences
inferred_recombinant_sequences = np.array(inferred_recombinant_sequences)
inferred_recombinant_sequences_flattened = inferred_recombinant_sequences.flatten()

# Position-by-position accuracy
accuracy = []
for i in range(n_cases):
	accuracy.append(np.mean(true_recombinant_sequences[i] == inferred_recombinant_sequences[i]))
accuracy = np.array(accuracy)
overall_accuracy = np.mean(accuracy)

boot_overall_accuracy = []
for _ in range(500):
	sampled_indices = np.random.choice(n_cases, size=n_cases, replace=True)  
	boot_overall_accuracy.append(np.mean(accuracy[sampled_indices]))

ci_lower_overall_accuracy, ci_upper_overall_accuracy = np.percentile(boot_overall_accuracy, [2.5, 97.5])

print(f"Position-by-position accuracy for all simulated recombinant sequences: {overall_accuracy:.3f} (95% CI: [{ci_lower_overall_accuracy:.3f}, {ci_upper_overall_accuracy:.3f}])")

""" SECTION: Calculate Position-By-Position Accuracy (Controls) """
print()
print("SECTION: Calculate Position-By-Position Accuracy (Controls)")

# Get true lineage per control from metadata
true_lineages_controls = [
    ref[ref['ID'] == id_[0]]['collapsed'].values[0]
    for id_ in sampled_seq_controls
]

# Generate "true" sequences (uniform sequence of same lineage)
true_control_sequences = np.array([[lin] * N for lin in true_lineages_controls])

# Generate inferred control sequences
inferred_control_sequences = []
for v in res_controls.values():
    if len(v[0]) == 1:
        inferred_control_sequences.append(v[0] * N)
    else:
        temp = [v[0][0]] * N
        for seg in v:
            temp[seg[1]:] = [seg[2]] * (N - seg[1])
        inferred_control_sequences.append(temp)
inferred_control_sequences = np.array(inferred_control_sequences)

# Compute per-sequence accuracy
accuracy_controls = np.mean(true_control_sequences == inferred_control_sequences, axis=1)
overall_accuracy_controls = accuracy_controls.mean()

# Bootstrap confidence interval
boot_control_accuracy = [
    np.mean(np.random.choice(accuracy_controls, size=n_controls, replace=True))
    for _ in range(500)
]
ci_lower_ctrl, ci_upper_ctrl = np.percentile(boot_control_accuracy, [2.5, 97.5])

print(f"Position-by-position accuracy for controls: {overall_accuracy_controls:.3f} (95% CI: [{ci_lower_ctrl:.3f}, {ci_upper_ctrl:.3f}])")

""" SECTION: Calculate Number of Matching Pango Lineages """
print()
print("SECTION: Calculate Number of Matching Pango Lineages")

# Get inferred set of Pango lineages for recombinants
inferred_pango_set_list = []
for res_breakpnt in res.values():
	if len(res_breakpnt[0]) == 1:
		inferred_pango_set_list.append(set(res_breakpnt[0]))
	else:
		temp = np.array(res_breakpnt).flatten()
		temp = {x for x in temp if isinstance(x, str) and not x.isnumeric()}
		inferred_pango_set_list.append(set(temp))

# ----------------------------
# 0.  Helper: canonicalise a combo
#      → alphabetically ordered tuple
# ----------------------------
def canonical(pair):
    return tuple(sorted(pair))        # ('BA.1', 'BA.2') always, never reversed

true_combo_tuples     = list(map(canonical, true_pango_set_list))
inferred_combo_sets   = list(map(frozenset, inferred_pango_set_list))
true_combo_sets       = list(map(frozenset, true_combo_tuples))

N = len(true_combo_tuples)
assert N == len(inferred_combo_sets), "true & inferred lists differ in length"

# --- CI helper ---------------------------------------------------------
def add_ci_columns(df, denom, count_col="count"):
    """
    Append columns ci_low / ci_high to `df`.
    `denom` can be a single integer or a vector with one entry per row.
    """
    # expand scalar denom to a vector if needed
    if np.isscalar(denom):
        denom = np.repeat(denom, len(df))

    ci_bounds = [proportion_confint(k, n, method="beta")
                 for k, n in zip(df[count_col], denom)]
    df[["ci_low", "ci_high"]] = pd.DataFrame(ci_bounds, index=df.index)
    return df

def match_type(t_set, inf_set):
    """
    Return 'perfect'  if the sets are identical;
           'partial'  if they overlap;
           'none'     otherwise.
    """
    if t_set == inf_set:
        return "perfect"
    elif t_set & inf_set:        # non-empty intersection
        return "partial"
    else:
        return "none"

# ----------------------------
# 1.  Proportion of TRUE combos
# ----------------------------
combo_counts = Counter(true_combo_tuples)
combo_df = (
    pd.DataFrame(combo_counts.items(), columns=["true_combo", "count"])
      .assign(prop=lambda d: d["count"] / N)
      .sort_values("prop", ascending=False)
      .reset_index(drop=True)
)
combo_df = add_ci_columns(combo_df, N)  

# ----------------------------
# 2.  Match type overall
# ----------------------------
match_types = [match_type(t, i) for t, i in zip(true_combo_sets,
                                                inferred_combo_sets)]

# original overall table (strict)
match_counter = Counter(match_types)

match_counter_incl = match_counter.copy()
match_counter_incl["partial"] += match_counter_incl.get("perfect", 0)

match_df_incl = (
    pd.DataFrame(match_counter_incl.items(), columns=["match", "count"])
      .assign(prop=lambda d: d["count"] / N)
)
match_df_incl = add_ci_columns(match_df_incl, N)

# ----------------------------
# 3.  Stratified (only PERFECT matches)
# ----------------------------
records = []
for combo, k in combo_counts.items():
    idx      = [c == combo for c in true_combo_tuples]
    perfect  = sum(m == "perfect" for m, flag in zip(match_types, idx) if flag)
    records.append((combo, k, perfect))

strat_perfect_df = (
    pd.DataFrame(records, columns=["true_combo", "n_combo", "count"])
      .assign(prop=lambda d: d["count"] / d["n_combo"])
)
strat_perfect_df = add_ci_columns(
    strat_perfect_df,
    strat_perfect_df["n_combo"],       # row-specific denominators
    count_col="count"
).sort_values("prop", ascending=False).reset_index(drop=True)

# ---  filter ----------------------------------------------------------
keep = strat_perfect_df["n_combo"] >= 10
tbl = strat_perfect_df.loc[keep].copy()

# ----------------------------
# 4.  Display
# ----------------------------
print("\nProportion of each *true* lineage combination "
      "(alphabetical order, sorted by proportion):")
print(combo_df.to_string(index=False))

print("\nOverall match quality  (partial = partial + perfect):")
print(match_df_incl.sort_values("prop", ascending=False)
                   .to_string(index=False))

print("\nPerfect-match accuracy *by true combination* "
      "(sorted by proportion):")
print(strat_perfect_df.to_string(index=False))

print(tbl.to_string(index=False))

from ast import literal_eval

# --- 1) Set centered LaTeX column names ---
headers = [
    r"\multicolumn{1}{c}{True lineages}",
    r"\multicolumn{1}{c}{Num. samples}",
    r"\multicolumn{1}{c}{Recovered}",
    r"\multicolumn{1}{c}{Prop.}",
    r"\multicolumn{1}{c}{2.5\,\% CI}",
    r"\multicolumn{1}{c}{97.5\,\% CI}",
]
assert tbl.shape[1] == len(headers), "Header count must match number of columns."
tbl.columns = headers

# --- 2) Formatters ---
def fmt_pair(x):
    # Render ('BA.5.2','BQ.1.1') as (BA.5.2, BQ.1.1)
    if isinstance(x, (tuple, list)):
        return f"({', '.join(map(str, x))})"
    if isinstance(x, str) and x.startswith("(") and "'" in x:
        try:
            t = literal_eval(x)
            if isinstance(t, (tuple, list)):
                return f"({', '.join(map(str, t))})"
        except Exception:
            pass
        return x.replace("'", "")
    return str(x)

def fmt_num(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    if isinstance(x, (float, np.floating)):
        return f"{int(round(x))}" if np.isclose(x, round(x)) else f"{x:.3f}"
    return str(x)

formatters = {col: fmt_num for col in tbl.columns}
formatters[headers[0]] = fmt_pair  # first column uses tuple-formatter

# --- 3) Write LaTeX ---
latex = tbl.to_latex(
    index=False,
    escape=False,                  # needed to keep \multicolumn and \% intact
    na_rep="",
    column_format='@{}lrrrrr@{}',  # = \begin{tabular}{@{}lrrrrr@{}}
    formatters=formatters
)

with open("output/table_strat_perfect.tex", "w") as f:
    f.write(latex)

""" SECTION: Calculate Number of Matching Pango Lineages (Controls) """
print()
print("SECTION: Calculate Number of Matching Pango Lineages (Controls)")

# Get true set of Pango lineages for controls		
true_pango_set_list_controls = []
for id_ in sampled_seq_controls:
    true_pango_set = []
    # Filter rows matching the ID in the tuple, and get the collapsed Pango lineage
    true_pango_set.extend(ref[ref['ID'] == id_[0]]['collapsed'].tolist())
    true_pango_set_list_controls.append(true_pango_set)

# Get inferred set of Pango lineages for controls
inferred_pango_set_list_controls = []
for res_breakpnt in res_controls.values():
	if len(res_breakpnt[0]) == 1:
		inferred_pango_set_list_controls.append(set(res_breakpnt[0]))
	else:
		temp = np.array(res_breakpnt).flatten()
		temp = {x for x in temp if isinstance(x, str) and not x.isnumeric()}
		inferred_pango_set_list_controls.append(set(temp))

# ------------------------------------------------------------------
# 0.  Helper: add 95 % Clopper–Pearson CIs to a DataFrame
# ------------------------------------------------------------------
def add_ci_columns(df, denom, count_col="count"):
    """
    Append ci_low / ci_high columns.
    `denom` can be a scalar (same N for all rows) or a 1-D array per row.
    """
    if np.isscalar(denom):
        denom = np.repeat(denom, len(df))

    ci_bounds = [proportion_confint(k, n, method="beta")
                 for k, n in zip(df[count_col], denom)]
    df[["ci_low", "ci_high"]] = pd.DataFrame(ci_bounds, index=df.index)
    return df


# ------------------------------------------------------------------
# 1.  Prepare data  -------------------------------------------------
#     • true_lineages: list[str] (single lineage each)
#     • inferred_sets: list[frozenset]
# ------------------------------------------------------------------
true_lineages  = [lst[0] for lst in true_pango_set_list_controls]      # flatten
inferred_sets  = list(map(frozenset, inferred_pango_set_list_controls))

N_controls = len(true_lineages)
assert N_controls == len(inferred_sets), "true vs inferred length mismatch"


# ------------------------------------------------------------------
# 2.  Proportion of EACH true lineage
# ------------------------------------------------------------------
lineage_counts = Counter(true_lineages)
prop_df = (
    pd.DataFrame(lineage_counts.items(), columns=["true_lineage", "count"])
      .assign(prop=lambda d: d["count"] / N_controls)
      .sort_values("prop", ascending=False)
      .reset_index(drop=True)
)
prop_df = add_ci_columns(prop_df, N_controls)


# ------------------------------------------------------------------
# 3.  Match-quality classification
# ------------------------------------------------------------------
def match_type_single(true_lin, inf_set):
    """
    Returns 'perfect'  – inferred set contains exactly {true_lin}
            'partial'  – inferred set contains true_lin but also others
            'none'     – inferred set lacks true_lin entirely
    """
    if inf_set == {true_lin}:
        return "perfect"
    elif true_lin in inf_set:
        return "partial"
    else:
        return "none"

match_types = [match_type_single(t, s) for t, s in zip(true_lineages, inferred_sets)]
# original overall table (strict)
match_counter = Counter(match_types)
# -----------------------------------------------------------------
# NEW: inclusive definition  (partial := partial + perfect)
# -----------------------------------------------------------------
match_counter_incl = match_counter.copy()
match_counter_incl["partial"] += match_counter_incl.get("perfect", 0)

match_df_incl = (
    pd.DataFrame(match_counter_incl.items(), columns=["match", "count"])
      .assign(prop=lambda d: d["count"] / N)
)
match_df_incl = add_ci_columns(match_df_incl, N)


# ------------------------------------------------------------------
# 4.  Stratified by true lineage  (perfect matches only)
# ------------------------------------------------------------------
records = []
for lin, k in lineage_counts.items():
    idx      = [t == lin for t in true_lineages]
    perfect  = sum((m == "perfect") for m, f in zip(match_types, idx) if f)
    records.append((lin, k, perfect))        # store counts, add CI later

strat_df = (
    pd.DataFrame(records,
                 columns=["true_lineage", "n_lineage", "count"])
      .assign(prop=lambda d: d["count"] / d["n_lineage"])
)
strat_df = add_ci_columns(
    strat_df,
    strat_df["n_lineage"],        # denominator varies by row
    count_col="count"
).sort_values("prop", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# 5.  Display
# ------------------------------------------------------------------
print("\nProportion of each *true* lineage (controls):")
print(prop_df.to_string(index=False))

print("\nOverall match quality  (partial = partial + perfect):")
print(match_df_incl.sort_values("prop", ascending=False)
                   .to_string(index=False))

print("\nPerfect-match accuracy BY true lineage (controls):")
print(strat_df.to_string(index=False))

""" SECTION: Match Rate vs. Hamming for Cases """

# ---------------------- 1.  read .zst FASTA ----------------------
fasta_zst = Path("../data/pango-consensus-sequences_genome-nuc.fasta.zst")       # ← your file

with fasta_zst.open("rb") as fh:
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    text_handle = io.TextIOWrapper(stream_reader, encoding="utf-8")
    records = list(SeqIO.parse(text_handle, "fasta"))

seqs = {rec.id: str(rec.seq).upper() for rec in records}
seq_len = {len(s) for s in seqs.values()}
if len(seq_len) != 1:
    raise ValueError("Not all consensus sequences have the same length")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Your distance function (kept verbatim)
# ──────────────────────────────────────────────────────────────────────────────
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return sum(
        c1 != c2
        for c1, c2 in zip(s1, s2)
        if c1 != "N" and c2 != "N"
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Extract the lineage pairs we actually need
#     * Drop duplicates
#     * Normalise order so linA < linB alphabetically
# ──────────────────────────────────────────────────────────────────────────────
# Example: strat_perfect_df already exists in memory --------------------------
# If the column stores the pair as a literal string like "('AY.4','AY.5')",
# use ast.literal_eval to convert each entry to a tuple.
pairs_series = (
    strat_perfect_df["true_combo"]
      .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple(x))
      .apply(lambda t: tuple(sorted(t)))       # alphabetical order
      .drop_duplicates()
)

# ---------------------------------------------------------------------------
# 4.  Compute distances only for these pairs
# ---------------------------------------------------------------------------
results = []
missing = set()                                # to report absent lineages

for linA, linB in pairs_series:
    try:
        seqA, seqB = seqs[linA], seqs[linB]
    except KeyError as e:
        missing.add(e.args[0])
        continue
    dist = hamming_distance(seqA, seqB)
    results.append((linA, linB, dist))

if missing:
    print(f"⚠️  {len(missing)} lineages listed in true_combo "
          f"not found in the FASTA: {sorted(missing)[:5]} ...")

dist_df = pd.DataFrame(results, columns=["linA", "linB", "dist"])
print(dist_df.head())

dist_df["true_combo"] = list(map(tuple, dist_df[["linA", "linB"]].values))

# keep just the join key and the metric you want to add
dist_keep = dist_df[["true_combo", "dist"]]

# ------------------------------------------------------------------
# 2.  Left-join onto the original table
# ------------------------------------------------------------------
strat_perfect_with_dist = (
    strat_perfect_df                       # or tbl if you prefer
        .merge(dist_keep,
               on="true_combo",            # join key
               how="left")                 # keep every row from the left table
)

df = (
    strat_perfect_with_dist
      .query("n_combo > 9")      # filter on sample size
      .dropna(subset=["dist"])    # drop rows without a distance
      .copy()
)

print(df)

# Data for plot
x = df["dist"].astype(float)
y = df["prop"].astype(float)
yerr = [y - df["ci_low"].astype(float), df["ci_high"].astype(float) - y]

# Presentation-friendly aesthetics
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

plt.figure(figsize=(10, 7))
plt.errorbar(
    x, y,
    yerr=yerr,
    fmt="o",
    capsize=5,
    elinewidth=1.5,
    markeredgewidth=1.5,
)
plt.xlabel("Hamming distance between lineage consensus sequences")
plt.ylabel("Correct proportion")
# plt.title("Perfect-match accuracy vs. genetic distance\n(only combos with ≥10 observations)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save (300 dpi for crisp slides)
plt.savefig("figs/perfect_match_vs_distance_filtered.png",
            dpi=300, bbox_inches="tight")
plt.close()