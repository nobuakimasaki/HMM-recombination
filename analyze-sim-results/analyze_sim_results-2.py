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
import statsmodels.api as sm

# Presentation-friendly aesthetics
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})

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

""" SECTION: Calculate Breakpoint Distances """ 

# Get inferred breakpoints and the lineages to the left and right
inferred_breakpoints_and_lineages = [res_breakpnt for res_breakpnt in res.values()]

# Do separately for one/two breakpoints
print()
print("SECTION: Calculate Breakpoint Distances")

inferred_breakpoints_list = []
for sublist in inferred_breakpoints_and_lineages:
    if len(sublist[0]) == 1:
        inferred_breakpoints_list.append([])
    else:
        temp = []
        for subsublist in sublist:
            temp.append(subsublist[1])
        inferred_breakpoints_list.append(temp)

breakpoint_distances_single = []
breakpoint_distances_double = []
match_number_1 = 0
match_number_2 = 0

# Calculate distances
for i in range(n_cases):
    true_bps = np.sort(np.asarray(breakpoints[i]))
    inf_bps  = np.sort(np.asarray(inferred_breakpoints_list[i]))

    if len(true_bps) == len(inf_bps) == 1:
        match_number_1 += 1
        d = np.abs(inf_bps[0] - true_bps[0])
        breakpoint_distances_single.append(float(d))

    elif len(true_bps) == len(inf_bps) == 2:
        match_number_2 += 1
        # Pair by 5'→3' order and take the per-recombinant mean
        d = np.abs(inf_bps - true_bps)          # elementwise: [|hat1-t1|, |hat2-t2|]
        breakpoint_distances_double.append(float(d.mean()))

boot_dist_single = []
boot_dist_double = []

for _ in range(500):
    sampled_single = np.random.choice(breakpoint_distances_single, size=len(breakpoint_distances_single), replace=True)  
    boot_dist_single.append(np.mean(sampled_single))

    sampled_double = np.random.choice(breakpoint_distances_double, size=len(breakpoint_distances_double), replace=True)  
    boot_dist_double.append(np.mean(sampled_double))

# Compute 95% confidence intervals
ci_single = np.percentile(boot_dist_single, [2.5, 97.5])
ci_double = np.percentile(boot_dist_double, [2.5, 97.5])

# Print point estimates and intervals
print(f"Single breakpoint mean distance: {np.mean(breakpoint_distances_single):.1f} (95% CI: {ci_single[0]:.1f}–{ci_single[1]:.1f})")
print(f"Double breakpoint mean distance: {np.mean(breakpoint_distances_double):.1f} (95% CI: {ci_double[0]:.1f}–{ci_double[1]:.1f})")

fig, ax = plt.subplots(figsize=(6, 4))

# Matplotlib violin plot (kernel-density estimate per group)
parts = ax.violinplot(
    [breakpoint_distances_single, breakpoint_distances_double],
    positions=[1, 2],                # x-coords
    showmeans=True,                  # horizontal bars at group means
    showmedians=False,               # medians hidden (means already shown)
    widths=0.8,                      # make violins wider/thinner
)

# Optional: customise fill & edge colours
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')      # default blue; tweak if you like
    pc.set_alpha(0.5)
    pc.set_edgecolor('k')

# Means appear as white bars by default; make them clearer
parts['cmeans'].set_edgecolor('k')
parts['cmeans'].set_linewidth(1.2)

# Axis labelling
ax.set_xticks([1, 2])
ax.set_xticklabels(["Single breakpoint", "Double breakpoint"])
ax.set_ylabel("(Average) breakpoint distance")
# ax.set_title("Distribution of breakpoint distances")

plt.tight_layout()
plt.savefig('figs/breakpoint_violin.png', dpi=100, bbox_inches='tight')
plt.close(fig)       # frees memory if you create many plots

""" SECTION: Confusion Matrix """
print()
print("SECTION: Confusion Matrix")

# Confusion matrix: true vs inferred number of breakpoints
true_vs_inferred_counts = []

for i in range(n_cases):
    true_count = len(breakpoints[i])
    inferred_count = len(inferred_breakpoints_list[i])
    true_vs_inferred_counts.append((true_count, inferred_count))

# Count occurrences of each (true, inferred) pair
conf_matrix_counts = Counter(true_vs_inferred_counts)

# Convert to DataFrame with correct column names
conf_matrix_df = pd.DataFrame.from_dict(conf_matrix_counts, orient='index', columns=['Count']).reset_index()
conf_matrix_df.columns = ['True_Breakpoints_Inferred_Breakpoints', 'Count']

# Split the tuple column into two separate columns
conf_matrix_df[['True_Breakpoints', 'Inferred_Breakpoints']] = pd.DataFrame(conf_matrix_df['True_Breakpoints_Inferred_Breakpoints'].tolist(), index=conf_matrix_df.index)
conf_matrix_df.drop(columns='True_Breakpoints_Inferred_Breakpoints', inplace=True)

# Pivot to make it look like a confusion matrix
conf_matrix_pivot = conf_matrix_df.pivot(index='True_Breakpoints', columns='Inferred_Breakpoints', values='Count').fillna(0).astype(int)

# Save
# conf_matrix_pivot.to_csv("output/results/breakpoint_confusion_matrix.csv")

# Print as formatted table
print("\nConfusion Matrix: True vs Inferred Breakpoints")
print(conf_matrix_pivot.to_string())

# Also print match counts if you'd like
print(f"\nExact matches for single-breakpoint cases: {match_number_1}")
print(f"Exact matches for double-breakpoint cases: {match_number_2}")

print()
print("SECTION: Edited Nucleotides (True Recombinant vs. Parents)")

# Get parental nucleotide sequences
true_seq_pairs = []
for id_pair in sampled_seq:
    s1 = ref[ref['ID'] == id_pair[0]]['Trimmed'].values[0]
    s2 = ref[ref['ID'] == id_pair[1]]['Trimmed'].values[0]
    true_seq_pairs.append((s1, s2))

# ------------------------------------------------------------
# Hamming distance *between* the two parental lineages
# ------------------------------------------------------------
parent_dists = np.array([
    hamming_distance(s1, s2)          # 1 line per recombinant
    for s1, s2 in true_seq_pairs
])

print("Hamming distance between parental sequences:")
print(parent_dists)

# If you also want a summary with 95 % CIs for the mean distance:
mean_parent_dist = parent_dists.mean()

# bootstrap CI (numeric data → bootstrap is still standard)
boot_means = [
    np.mean(np.random.choice(parent_dists, size=len(parent_dists), replace=True))
    for _ in range(500)
]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

print(f"Mean parent-parent Hamming distance: {mean_parent_dist:.1f} "
      f"[95 % CI {ci_low:.1f}, {ci_high:.1f}]")

""" SECTION: Edited Nucleotides vs. Sensitivity """
print()
print("SECTION: Edited Nucleotides vs. Sensitivity")

cases_positive_index_breakpnt = np.zeros(n_cases, dtype=int)
# Obtain vector of 0s and 1s based on whether we predicted the sequence as a recombinant
for i, breakpnt in enumerate(res.values()):
    # If the first entry is not length 1 (meaning there is a breakpoint)
    if len(breakpnt[0]) != 1:
        cases_positive_index_breakpnt[i] = 1

# ------------------------------------------------------------
# 1.  Fit the logit model  (parental distance → detection)
# ------------------------------------------------------------
y  = cases_positive_index_breakpnt.astype(int)          # 0/1, no NaNs
X  = sm.add_constant(parent_dists)                      # intercept + distance
res = sm.Logit(y, X).fit(disp=False)                    # suppress console spam

print(res.summary())      # R-style table with β, SE, z, p, 95 % CI
sum2 = res.summary2()          # richer than summary()

# tables[1] is the coefficients table
coef_tbl = sum2.tables[1].copy()      # DataFrame already

# pretty-print with scientific notation or any format you like
print(coef_tbl.to_string(float_format="{:.6g}".format))
# ------------------------------------------------------------
# 2.  Prediction grid + universal extraction of columns
# ------------------------------------------------------------
x_grid   = np.linspace(parent_dists.min(), parent_dists.max(), 100)
X_grid   = sm.add_constant(x_grid)

pr       = res.get_prediction(X_grid)                   # PredictionResults
sf       = pr.summary_frame(alpha=0.05)                 # DataFrame of extras

# --- column names differ across versions ----------------------------------
col_mean  = next(c for c in sf if "mean"   in c.lower() or "pred"  in c.lower())
col_lower = next(c for c in sf if "lower"  in c.lower())
col_upper = next(c for c in sf if "upper"  in c.lower())

y_hat     = sf[col_mean ].to_numpy()        # fitted probability
y_lower   = sf[col_lower].to_numpy()        # 95 % Wald band (lower)
y_upper   = sf[col_upper].to_numpy()        # 95 % Wald band (upper)

# parent_dists: np.array of distances; y: 0/1 array
pd.DataFrame({"parent_dist": parent_dists, "detected": y}).to_csv(
    "output/parent_dist_and_detection.csv", index=False
)

# ------------------------------------------------------------
# 3.  Plot
# ------------------------------------------------------------
# Presentation-friendly aesthetics
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})

plt.scatter(parent_dists, y, s=20, alpha=0.1, label="Simulated recombinants")

# no legend entry for the fitted curve
plt.plot(x_grid, y_hat, color="red", label="_nolegend_")

plt.fill_between(x_grid, y_lower, y_upper, color="red", alpha=0.1,
                 label="95 % Wald CI")

plt.xlabel("Parental Hamming distance")
plt.ylabel("Sensitivity")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
plt.tight_layout()
plt.savefig("figs/sens_vs_parentdist.png", dpi=120)

# ------------------------------------------------------------
# 4.  Programmatic slope & CI
# ------------------------------------------------------------
ci_arr = res.conf_int(alpha=0.05)   # 2-column array or DataFrame

if isinstance(ci_arr, np.ndarray):          # array: shape (p, 2)
    ci_lower, ci_upper = ci_arr[1]          # row 1 = slope for x1
else:                                       # DataFrame
    ci_lower, ci_upper = ci_arr.iloc[1]

b1 = res.params[1]                          # slope estimate
print(f"Slope β₁ = {b1:.3f}  (95 % CI [{ci_lower:.3f}, {ci_upper:.3f}])")
# res  is your LogitResults

coef_tbl = res.summary2().tables[1].copy()      # already a DataFrame

# Add exponentiated columns
coef_tbl["OR"]        = np.exp(coef_tbl["Coef."])
coef_tbl["CI_low_OR"] = np.exp(coef_tbl["[0.025"])
coef_tbl["CI_high_OR"]= np.exp(coef_tbl["0.975]"])

# Nicely formatted printout
print(
    coef_tbl[["OR", "CI_low_OR", "CI_high_OR"]]
        .applymap("{:.3g}".format)
        .rename(columns={"CI_low_OR":"2.5 %", "CI_high_OR":"97.5 %"}))