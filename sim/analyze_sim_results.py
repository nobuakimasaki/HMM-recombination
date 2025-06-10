import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
# import random
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import Counter
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
# from reportlab.lib import colors

# Set seed for reproducibility
np.random.seed(108)

# Load results
optim_res = pd.read_csv("output/inferred/optimization_results.csv")
optim_res_controls = pd.read_csv("output/inferred/optimization_results_controls.csv")

with open("output/inferred/inferred_lineages.json", "r") as file:
    res = json.load(file)
with open("output/inferred/inferred_lineages_controls.json", "r") as file:
    res_controls = json.load(file)

# Load reference set
ref = pd.read_csv("output/sliding_windows/seq_pango_sorted_2022-11-19_2022-12-18.csv.gz")
# Load sampled IDs and breakpoints
sampled_seq = pd.read_csv("output/simulated_sequences/sampled_sequences.csv", header = None).values.tolist()
sampled_seq_controls = pd.read_csv("output/simulated_sequences/sampled_sequences_control.csv", header = None).values.tolist()
breakpoints = pd.read_csv("output/simulated_sequences/breakpoints.csv", header = None).values.tolist()
# Remove 'nan' values from sublists
breakpoints = [[int(x) for x in sublist if not (isinstance(x, float) and np.isnan(x))] for sublist in breakpoints]

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
	#print(breakpnt)
	# If the first entry is not length 1 (meaning there is a breakpoint)
	if len(breakpnt[0]) != 1:
		cases_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		cases_positive_index_breakpnt[i] = 0

sens_breakpnt = sum(cases_positive_index_breakpnt)/n_cases

# Obtain vector of 0s and 1s based on whether we predicted the sequence as a recombinant
for i, breakpnt in enumerate(res_controls.values()):
	# If the first entry is not length 1 (meaning there is a breakpoint)
	if len(breakpnt[0]) != 1:
		controls_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		controls_positive_index_breakpnt[i] = 0

spec_breakpnt = sum(controls_positive_index_breakpnt == 0)/n_controls

# cases_positive_index_s = optim_res['est_s'] < 1
# controls_positive_index_s = optim_res_controls['est_s'] < 1

# sens_s = sum(cases_positive_index_s)/n_cases
# spec_s = sum(controls_positive_index_s == 0)/n_controls

# Bootstrap the vector of 0s and 1s to get a bootstrap percentile interval for sensitivity and specificity
boot_cases_breakpnt_list = []
boot_controls_breakpnt_list = []
# boot_cases_s_list = []
# boot_controls_s_list = []

for _ in range(500):
    boot_cases_breakpnt = np.random.choice(cases_positive_index_breakpnt, size=n_cases, replace=True)
    boot_controls_breakpnt = np.random.choice(controls_positive_index_breakpnt, size=n_controls, replace=True)
    boot_cases_breakpnt_list.append(boot_cases_breakpnt)
    boot_controls_breakpnt_list.append(boot_controls_breakpnt)
# for _ in range(500):
#     boot_cases_s = np.random.choice(cases_positive_index_s, size=n_cases, replace=True)
#     boot_controls_s = np.random.choice(controls_positive_index_s, size=n_controls, replace=True)
#     boot_cases_s_list.append(boot_cases_s)
#     boot_controls_s_list.append(boot_controls_s)

boot_sens_breakpnt = [sum(positive_index)/n_cases for positive_index in boot_cases_breakpnt_list]
boot_spec_breakpnt = [sum(positive_index == 0)/n_controls for positive_index in boot_controls_breakpnt_list]
# boot_sens_s = [sum(positive_index)/n_cases for positive_index in boot_cases_s_list]
# boot_spec_s = [sum(positive_index == 0)/n_controls for positive_index in boot_controls_s_list]

# Calculate bootstrap percentile intervals
ci_lower_sens_breakpnt, ci_upper_sens_breakpnt = np.percentile(boot_sens_breakpnt, [2.5, 97.5])
ci_lower_spec_breakpnt, ci_upper_spec_breakpnt = np.percentile(boot_spec_breakpnt, [2.5, 97.5])
# ci_lower_sens_s, ci_upper_sens_s = np.percentile(boot_sens_s, [2.5, 97.5])
# ci_lower_spec_s, ci_upper_spec_s = np.percentile(boot_spec_s, [2.5, 97.5])

print(f"Sensitivity using breakpoint: {sens_breakpnt:.3f} (95% CI: [{ci_lower_sens_breakpnt:.3f}, {ci_upper_sens_breakpnt:.3f}])")
print(f"Specificity using breakpoint: {spec_breakpnt:.3f} (95% CI: [{ci_lower_spec_breakpnt:.3f}, {ci_upper_spec_breakpnt:.3f}])")

# print(f"Sensitivity using estimated transition probabilities: {sens_s:.3f} (95% CI: [{ci_lower_sens_s:.3f}, {ci_upper_sens_s:.3f}])")
# print(f"Specificity using estimated transition probabilities: {spec_s:.3f} (95% CI: [{ci_lower_spec_s:.3f}, {ci_upper_spec_s:.3f}])")

""" SECTION: Calculate Predicted Proportion of Recombinants """  
print()
print("SECTION: Calculate Predicted Proportion of Recombinants")
prop_breakpnt = (sum(cases_positive_index_breakpnt) + sum(controls_positive_index_breakpnt))/(n_cases + n_controls)
# prop_s = (sum(cases_positive_index_s) + sum(controls_positive_index_s))/(n_cases + n_controls)

boot_prop_breakpnt = []
# boot_prop_s = []
for i in range(len(boot_cases_breakpnt_list)):
	boot_prop_breakpnt.append((sum(boot_cases_breakpnt_list[i]) + sum(boot_controls_breakpnt_list[i]))/(n_cases + n_controls))
# for i in range(len(boot_cases_s_list)):
# 	boot_prop_s.append((sum(boot_cases_s_list[i]) + sum(boot_controls_s_list[i]))/(n_cases + n_controls))

# Calculate bootstrap percentile intervals
ci_lower_prop_breakpnt, ci_upper_prop_breakpnt = np.percentile(boot_prop_breakpnt, [2.5, 97.5])
# ci_lower_prop_s, ci_upper_prop_s = np.percentile(boot_prop_s, [2.5, 97.5])

print(f"Predicted proportion of recombinants using breakpoint: {prop_breakpnt:.3f} (95% CI: [{ci_lower_prop_breakpnt:.3f}, {ci_upper_prop_breakpnt:.3f}])")
# print(f"Predicted proportion of recombinants using s: {prop_s:.3f} (95% CI: [{ci_lower_prop_s:.3f}, {ci_upper_prop_s:.3f}])")

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

# === SECTION: Position-by-position accuracy for controls ===
print()
print("SECTION: Position-By-Position Accuracy (For Controls)")

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

# Compare true and inferred set for recombinants
match_list = []
partial_match_list = []
for i in range(n_cases):
	if set(true_pango_set_list[i]) == inferred_pango_set_list[i]:
		match_list.append(1)
		partial_match_list.append(1)
	elif bool(set(true_pango_set_list[i]) & inferred_pango_set_list[i]):
		match_list.append(0)
		partial_match_list.append(1)
	else:
		match_list.append(0)
		partial_match_list.append(0)

# Compare true and inferred set for controls
match_list_controls = []
partial_match_list_controls = []
for i in range(n_controls):
	if set(true_pango_set_list_controls[i]) == inferred_pango_set_list_controls[i]:
		match_list_controls.append(1)
		partial_match_list_controls.append(1)
	elif bool(set(true_pango_set_list_controls[i]) & inferred_pango_set_list_controls[i]):
		match_list_controls.append(0)
		partial_match_list_controls.append(1)
	else:
		match_list_controls.append(0)
		partial_match_list_controls.append(0)

match_list = np.array(match_list)
partial_match_list = np.array(partial_match_list)
match_list_controls = np.array(match_list_controls)
partial_match_list_controls = np.array(partial_match_list_controls)

match_prob = np.mean(match_list)
partial_match_prob = np.mean(partial_match_list)
match_prob_controls = np.mean(match_list_controls)
partial_match_prob_controls = np.mean(partial_match_list_controls)

boot_match_prob = []
boot_partial_match_prob = []
boot_match_prob_controls = []
boot_partial_match_prob_controls = []
for _ in range(500):
	sampled_indices_cases = np.random.choice(n_cases, size=n_cases, replace=True)  
	boot_match_prob.append(np.mean(match_list[sampled_indices_cases]))
	boot_partial_match_prob.append(np.mean(partial_match_list[sampled_indices_cases]))
	sampled_indices_controls = np.random.choice(n_controls, size=n_controls, replace=True)  
	boot_match_prob_controls.append(np.mean(match_list_controls[sampled_indices_controls]))
	boot_partial_match_prob_controls.append(np.mean(partial_match_list_controls[sampled_indices_controls]))

ci_lower_match_prob, ci_upper_match_prob = np.percentile(boot_match_prob, [2.5, 97.5])
ci_lower_partial_match_prob, ci_upper_partial_match_prob = np.percentile(boot_partial_match_prob, [2.5, 97.5])
ci_lower_match_prob_controls, ci_upper_match_prob_controls = np.percentile(boot_match_prob_controls, [2.5, 97.5])
ci_lower_partial_match_prob_controls, ci_upper_partial_match_prob_controls = np.percentile(boot_partial_match_prob_controls, [2.5, 97.5])

print(f"Proportion of matching lineages for recombinants: {match_prob:.3f} (95% CI: [{ci_lower_match_prob:.3f}, {ci_upper_match_prob:.3f}])")
print(f"Proportion of partially matching lineages for recombinants: {partial_match_prob:.3f} (95% CI: [{ci_lower_partial_match_prob:.3f}, {ci_upper_partial_match_prob:.3f}])")
print(f"Proportion of matching lineages for controls: {match_prob_controls:.3f} (95% CI: [{ci_lower_match_prob_controls:.3f}, {ci_upper_match_prob_controls:.3f}])")
print(f"Proportion of partially matching lineages for controls: {partial_match_prob_controls:.3f} (95% CI: [{ci_lower_partial_match_prob_controls:.3f}, {ci_upper_partial_match_prob_controls:.3f}])")

""" SECTION: Calculate Breakpoint Distances """ 

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
    true_bps = breakpoints[i]
    inferred_bps = inferred_breakpoints_list[i]

    if len(true_bps) == len(inferred_bps):

        if len(true_bps) == 1:
            # Just compute single absolute distance
            match_number_1 += 1
            breakpoint_distances_single.append(np.abs(inferred_bps[0] - true_bps[0]))

        elif len(true_bps) == 2:
            # Choose ordering that minimizes total absolute distance
            match_number_2 += 1
            dist1 = np.mean([np.abs(inferred_bps[0] - true_bps[0]), np.abs(inferred_bps[1] - true_bps[1])])
            dist2 = np.mean([np.abs(inferred_bps[1] - true_bps[0]), np.abs(inferred_bps[0] - true_bps[1])])
            if dist1 <= dist2:
                breakpoint_distances_double.append(dist1)
            else:
                breakpoint_distances_double.append(dist2)

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

# Create boxplot
plt.boxplot([breakpoint_distances_single, breakpoint_distances_double],
            labels=["Single breakpoint", "Double breakpoint"])
plt.ylabel("Distance between true and inferred breakpoint")
plt.title("Distribution of breakpoint distances")
plt.tight_layout()
plt.savefig('output/results/breakpoint_boxplot.png', dpi=100, bbox_inches='tight')
plt.clf()

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

# Load true recombinant sequences (encoded as integers)
true_recombinant_numeric = pd.read_csv("output/simulated_sequences/recombinants.csv", header=None).values

# Inverse mapping from integer to nucleotide
allele_order = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
int_to_nuc = {v: k for k, v in allele_order.items()}

# Convert numeric sequences to nucleotide strings
true_nuc_seqs = [
    ''.join([int_to_nuc[i] for i in row])
    for row in true_recombinant_numeric
]
#print(true_seq_pairs[1])
#print(true_nuc_seqs[1])

# Re-use true_seq_pairs (s1, s2 = parent nucleotide sequences)
edit_counts_true = []
for i in range(n_cases):
    s1, s2 = true_seq_pairs[i]
    recombinant = true_nuc_seqs[i]
    d1 = hamming_distance(recombinant, s1)
    d2 = hamming_distance(recombinant, s2)
    edit_counts_true.append(min(d1, d2))

edit_counts_true = np.array(edit_counts_true)
#mean_edit_true = np.mean(edit_counts_true)

# # Bootstrap confidence intervals
# boot_means_true = [
#     np.mean(np.random.choice(edit_counts_true, size=n_cases, replace=True))
#     for _ in range(500)
# ]
# ci_lower_true, ci_upper_true = np.percentile(boot_means_true, [2.5, 97.5])

print(f"Number of edited nucleotides (true recombinants): {edit_counts_true}")

""" SECTION: Edited Nucleotides vs. Sensitivity """
print()
print("SECTION: Edited Nucleotides vs. Sensitivity")

edit_counts = np.array(edit_counts_true).reshape(-1, 1)  # Predictor variable (X)

# Fit logistic regression model
model_edit = LogisticRegression()
model_edit.fit(edit_counts, cases_positive_index_breakpnt)

# Generate smooth predictions
x_smooth = np.linspace(edit_counts.min(), edit_counts.max(), 100).reshape(-1, 1)
y_smooth = model_edit.predict_proba(x_smooth)[:, 1]

# Coefficients
b0_edit, b1_edit = model_edit.intercept_[0], model_edit.coef_[0][0]

# Plot scatter
plt.scatter(edit_counts, cases_positive_index_breakpnt, marker='o', color='blue', s=20, alpha=0.1)

# Bootstrap CI for logistic curve
b1_edit_boot = []
y_pred_samples = np.zeros((500, len(x_smooth)))

for _ in range(500):
    indices = np.random.choice(len(edit_counts), size=len(edit_counts), replace=True)
    X_boot = edit_counts[indices]
    y_boot = np.array(cases_positive_index_breakpnt)[indices]
    
    model_boot = LogisticRegression()
    model_boot.fit(X_boot, y_boot)
    
    b1_edit_boot.append(model_boot.coef_[0][0])
    y_pred_samples[_, :] = model_boot.predict_proba(x_smooth)[:, 1]

# CI for predicted probabilities
y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# Plot logistic curve and CI
plt.plot(x_smooth, y_smooth, color='red', label="Predicted probability from logistic regression")
plt.fill_between(x_smooth.ravel(), y_lower, y_upper, color='red', alpha=0.2, label="95% CI")

# Add labels and formatting
plt.xlabel("Minimum number of edited nucleotides")
plt.ylabel("Probability of recombinant classification (sensitivity)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)

plt.tight_layout()
plt.savefig('output/results/sens_vs_edits.png', dpi=100, bbox_inches='tight')
plt.clf()

# Report coefficient CI
b1_edit_lower = np.percentile(b1_edit_boot, 2.5)
b1_edit_upper = np.percentile(b1_edit_boot, 97.5)

print(f"Estimated logistic regression slope (edits): {b1_edit:.3f} (95% CI: [{b1_edit_lower:.3f}, {b1_edit_upper:.3f}])")

# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(hamming_distances, match_list)

# # Generate smooth predictions
# x_smooth = np.linspace(min(hamming_distances), max(hamming_distances), 100).reshape(-1, 1)
# y_smooth = model.predict_proba(x_smooth)[:, 1] 

# # Get the beta coefficients (intercept and slope)
# beta0, beta1 = model.intercept_[0], model.coef_[0][0]

# # Create scatter plot with smaller, more transparent points
# plt.scatter(hamming_distances, match_list, marker='o', color='blue', s=20, alpha=0.1)

# # Bootstrap confidence intervals
# n_bootstraps = 500
# beta1_boot = []
# y_pred_samples = np.zeros((n_bootstraps, len(x_smooth)))

# for i in range(n_bootstraps):
#     # Resample the data
#     indices = np.random.choice(len(hamming_distances), len(hamming_distances), replace=True)
#     hamming_boot = hamming_distances[indices]
#     match_boot = np.array(match_list)[indices]
    
#     # Fit logistic regression to the resampled data
#     model_boot = LogisticRegression()
#     model_boot.fit(hamming_boot.reshape(-1, 1), match_boot)

#     # Store beta1 estimate
#     beta1_boot.append(model_boot.coef_[0][0])
    
#     # Predict probabilities for the smooth x values
#     y_pred_samples[i, :] = model_boot.predict_proba(x_smooth)[:, 1]

# # Compute the 95% confidence interval
# y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
# y_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# # Plot logistic regression probability curve
# plt.plot(x_smooth, y_smooth, color='red', label="Predicted probability from logistic regression fit")

# # Plot confidence band
# plt.fill_between(x_smooth.ravel(), y_lower, y_upper, color='red', alpha=0.2, label="95% CI")

# # Add labels and title
# plt.xlabel("Hamming distance between parental sequences")
# plt.ylabel("Correct lineages inferred?")
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)

# # Show the plot
# plt.tight_layout()

# # Save figure as PNG
# plt.savefig('output/analysis/correct_vs_hamming.png', dpi=100, bbox_inches='tight')

# # Clear the figure
# plt.clf()

# # Compute the 95% confidence interval for beta1
# beta1_lower = np.percentile(beta1_boot, 2.5)
# beta1_upper = np.percentile(beta1_boot, 97.5)

# print(f"Estimated logistic regression slope: {beta1} (95% CI: [{beta1_lower}, {beta1_upper}])")

""" SECTION: Hamming Distance vs. Sensitivity """
# print()
# print("SECTION: Hamming Distance vs. Sensitivity")

# # Fit logistic regression model
# model_sens = LogisticRegression()
# model_sens.fit(hamming_distances, cases_positive_index_s)

# # Generate smooth predictions
# x_smooth = np.linspace(min(hamming_distances), max(hamming_distances), 100).reshape(-1, 1)
# y_smooth = model_sens.predict_proba(x_smooth)[:, 1] 

# # Get the beta coefficients (intercept and slope)
# b0_sens, b1_sens = model_sens.intercept_[0], model_sens.coef_[0][0]

# # Create scatter plot with smaller, more transparent points
# plt.scatter(hamming_distances, cases_positive_index_s, marker='o', color='blue', s=20, alpha=0.1)

# # Bootstrap confidence intervals
# n_bootstraps = 500
# b1_sens_boot = []
# y_pred_samples = np.zeros((n_bootstraps, len(x_smooth)))

# for i in range(n_bootstraps):
#     # Resample the data
#     indices = np.random.choice(len(hamming_distances), len(hamming_distances), replace=True)
#     hamming_boot = hamming_distances[indices]
#     positive_boot = np.array(cases_positive_index_s)[indices]
    
#     # Fit logistic regression to the resampled data
#     model_sens_boot = LogisticRegression()
#     model_sens_boot.fit(hamming_boot.reshape(-1, 1), positive_boot)

#     # Store beta1 estimate
#     b1_sens_boot.append(model_sens_boot.coef_[0][0])
    
#     # Predict probabilities for the smooth x values
#     y_pred_samples[i, :] = model_sens_boot.predict_proba(x_smooth)[:, 1]

# # Compute the 95% confidence interval
# y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
# y_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# # Plot logistic regression probability curve
# plt.plot(x_smooth, y_smooth, color='red', label="Predicted probability from logistic regression fit")

# # Plot confidence band
# plt.fill_between(x_smooth.ravel(), y_lower, y_upper, color='red', alpha=0.2, label="95% CI")

# # Add labels and title
# plt.xlabel("Hamming distance between parental sequences")
# plt.ylabel("Classified as a recombinant?")
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)

# # Show the plot
# plt.tight_layout()

# # Save figure as PNG
# plt.savefig('output/results/sens_vs_hamming.png', dpi=100, bbox_inches='tight')

# # Clear the figure
# plt.clf()

# # Compute the 95% confidence interval for beta1
# b1_sens_lower = np.percentile(b1_sens_boot, 2.5)
# b1_sens_upper = np.percentile(b1_sens_boot, 97.5)

# print(f"Estimated logistic regression slope: {b1_sens:.3f} (95% CI: [{b1_sens_lower:.3f}, {b1_sens_upper:.3f}])")

""" SECTION: Plot First Few Recombinant Sequences and Inferred Lineages """
print()
print("SECTION: Plot First Few Recombinant Sequences and Inferred Lineage")

# Filter to first 20 sequences
recombinant_sequences_filt = true_recombinant_sequences[:20]
inferred_recombinant_sequences_filt = inferred_recombinant_sequences[:20]

# Get unique lineage labels from both sets
unique_lineages = sorted(set(lin for seq in list(recombinant_sequences_filt) + list(inferred_recombinant_sequences_filt) for lin in seq))
lineage_map = {lin: i for i, lin in enumerate(unique_lineages)}

# Convert to numeric matrices
recombinant_sequences_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in recombinant_sequences_filt])
inferred_recombinant_sequences_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in inferred_recombinant_sequences_filt])

# Create consistent colormap
num_colors = len(unique_lineages)
palette = sns.color_palette("husl", num_colors)
cmap = plt.cm.colors.ListedColormap(palette)

# Set consistent color scale bounds
vmin = 0
vmax = num_colors - 1

# Create figure
fig, ax = plt.subplots(1, 2, figsize=(13, 10))  # 1 row, 2 columns

# Plot 1: True lineages
im0 = ax[0].imshow(recombinant_sequences_matrix, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=vmin, vmax=vmax)
ax[0].set_xlabel("Position")
ax[0].set_ylabel("Sequence Index")
ax[0].set_title("True Lineages")
ax[0].invert_yaxis()
ax[0].tick_params(left=False, bottom=False)

# Plot 2: Inferred lineages
im1 = ax[1].imshow(inferred_recombinant_sequences_matrix, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=vmin, vmax=vmax)
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Sequence Index")
ax[1].set_title("Inferred Lineages")
ax[1].invert_yaxis()
ax[1].tick_params(left=False, bottom=False)

# Legend sorted by lineage index
legend_patches = [mpatches.Patch(color=cmap(i), label=lin) for lin, i in sorted(lineage_map.items(), key=lambda x: x[1])]
fig.legend(handles=legend_patches, title="Lineage", bbox_to_anchor=(1.05, 0.5), loc='center left')

# Finalize and save
plt.tight_layout()
plt.savefig('output/results/first_20_true_and_inferred.png', dpi=100, bbox_inches='tight')
plt.clf()
""" SECTION: Write Results to File """
# print()
# print("SECTION: Write Results to File")

# with open("output/results/sim_results.txt", "w") as f:
#     f.write(f"Number of cases: {round(n_cases)}\n")
#     f.write(f"Number of controls: {round(n_controls)}\n")
#     f.write(f"Length of trimmed genome: {round(N)}\n")
#     f.write(f"Number of Pango lineages after collapsing: {round(M)}\n")
#     f.write(f"Pango lineages after collapsing: {Pango_lineage_set}\n")

#     f.write("\nSECTION: Calculate Sensitivity and Specificity\n")
#     f.write(f"Sensitivity using breakpoint: {sens_breakpnt:.3f} (95% CI: [{ci_lower_sens_breakpnt:.3f}, {ci_upper_sens_breakpnt:.3f}])\n")
#     f.write(f"Specificity using breakpoint: {spec_breakpnt:.3f} (95% CI: [{ci_lower_spec_breakpnt:.3f}, {ci_upper_spec_breakpnt:.3f}])\n")
#     f.write(f"Sensitivity using estimated transition probabilities: {sens_s:.3f} (95% CI: [{ci_lower_sens_s:.3f}, {ci_upper_sens_s:.3f}])\n")
#     f.write(f"Specificity using estimated transition probabilities: {spec_s:.3f} (95% CI: [{ci_lower_spec_s:.3f}, {ci_upper_spec_s:.3f}])\n")

#     f.write("\nSECTION: Calculate Predicted Proportion of Recombinants\n")
#     f.write(f"Predicted proportion of recombinants using breakpoint: {prop_breakpnt:.3f} (95% CI: [{ci_lower_prop_breakpnt:.3f}, {ci_upper_prop_breakpnt:.3f}])\n")
#     f.write(f"Predicted proportion of recombinants using s: {prop_s:.3f} (95% CI: [{ci_lower_prop_s:.3f}, {ci_upper_prop_s:.3f}])\n")

#     f.write("\nSECTION: Calculate Position-By-Position Accuracy\n")
#     f.write(f"Position-by-position accuracy for all simulated recombinant sequences: {overall_accuracy:.3f} (95% CI: [{ci_lower_overall_accuracy:.3f}, {ci_upper_overall_accuracy:.3f}])\n")

#     f.write("\nSECTION: Calculate Number of Matching Pango Lineages\n")
#     f.write(f"Proportion of matching lineages for recombinants: {match_prob:.3f} (95% CI: [{ci_lower_match_prob:.3f}, {ci_upper_match_prob:.3f}])\n")
#     f.write(f"Proportion of partially matching lineages for recombinants: {partial_match_prob:.3f} (95% CI: [{ci_lower_partial_match_prob:.3f}, {ci_upper_partial_match_prob:.3f}])\n")
#     f.write(f"Proportion of matching lineages for controls: {match_prob_controls:.3f} (95% CI: [{ci_lower_match_prob_controls:.3f}, {ci_upper_match_prob_controls:.3f}])\n")
#     f.write(f"Proportion of partially matching lineages for controls: {partial_match_prob_controls:.3f} (95% CI: [{ci_lower_partial_match_prob_controls:.3f}, {ci_upper_partial_match_prob_controls:.3f}])\n")

#     f.write("\nSECTION: Calculate Breakpoint Distances\n")
#     f.write(f"Mean distance between true and inferred breakpoints: {round(np.mean(breakpoint_distances))}\n")

#     f.write("\nSECTION: Hamming Distance vs. Sensitivity\n")
#     f.write(f"Estimated logistic regression slope: {b1_sens:.3f} (95% CI: [{b1_sens_lower:.3f}, {b1_sens_upper:.3f}])\n")
