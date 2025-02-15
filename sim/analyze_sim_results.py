import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
# import random
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
ref = pd.read_csv("output/seq_pango_sorted.csv")
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
	# If the first entry is not length 1 (meaning there is no breakpoint)
	if len(breakpnt[0]) != 1:
		cases_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		cases_positive_index_breakpnt[i] = 0

sens_breakpnt = sum(cases_positive_index_breakpnt)/n_cases

# Obtain vector of 0s and 1s based on whether we predicted the sequence as a recombinant
for i, breakpnt in enumerate(res_controls.values()):
	# If the first entry is not length 1 (meaning there is no breakpoint)
	if len(breakpnt[0]) != 1:
		controls_positive_index_breakpnt[i] = 1
	if len(breakpnt[0]) == 1:
		controls_positive_index_breakpnt[i] = 0

spec_breakpnt = sum(controls_positive_index_breakpnt == 0)/n_controls

cases_positive_index_s = optim_res['est_s'] < 1
controls_positive_index_s = optim_res_controls['est_s'] < 1

sens_s = sum(cases_positive_index_s)/n_cases
spec_s = sum(controls_positive_index_s == 0)/n_controls

# Bootstrap the vector of 0s and 1s to get a bootstrap percentile interval for sensitivity and specificity
boot_cases_breakpnt_list = []
boot_controls_breakpnt_list = []
boot_cases_s_list = []
boot_controls_s_list = []

for _ in range(500):
    boot_cases_breakpnt = np.random.choice(cases_positive_index_breakpnt, size=n_cases, replace=True)
    boot_controls_breakpnt = np.random.choice(controls_positive_index_breakpnt, size=n_controls, replace=True)
    boot_cases_breakpnt_list.append(boot_cases_breakpnt)
    boot_controls_breakpnt_list.append(boot_controls_breakpnt)
for _ in range(500):
    boot_cases_s = np.random.choice(cases_positive_index_s, size=n_cases, replace=True)
    boot_controls_s = np.random.choice(controls_positive_index_s, size=n_controls, replace=True)
    boot_cases_s_list.append(boot_cases_s)
    boot_controls_s_list.append(boot_controls_s)

boot_sens_breakpnt = [sum(positive_index)/n_cases for positive_index in boot_cases_breakpnt_list]
boot_spec_breakpnt = [sum(positive_index == 0)/n_controls for positive_index in boot_controls_breakpnt_list]
boot_sens_s = [sum(positive_index)/n_cases for positive_index in boot_cases_s_list]
boot_spec_s = [sum(positive_index == 0)/n_controls for positive_index in boot_controls_s_list]

# Calculate bootstrap percentile intervals
ci_lower_sens_breakpnt, ci_upper_sens_breakpnt = np.percentile(boot_sens_breakpnt, [2.5, 97.5])
ci_lower_spec_breakpnt, ci_upper_spec_breakpnt = np.percentile(boot_spec_breakpnt, [2.5, 97.5])
ci_lower_sens_s, ci_upper_sens_s = np.percentile(boot_sens_s, [2.5, 97.5])
ci_lower_spec_s, ci_upper_spec_s = np.percentile(boot_spec_s, [2.5, 97.5])

print(f"Sensitivity using breakpoint: {sens_breakpnt:.3f} (95% CI: [{ci_lower_sens_breakpnt:.3f}, {ci_upper_sens_breakpnt:.3f}])")
print(f"Specificity using breakpoint: {spec_breakpnt:.3f} (95% CI: [{ci_lower_spec_breakpnt:.3f}, {ci_upper_spec_breakpnt:.3f}])")

print(f"Sensitivity using estimated transition probabilities: {sens_s:.3f} (95% CI: [{ci_lower_sens_s:.3f}, {ci_upper_sens_s:.3f}])")
print(f"Specificity using estimated transition probabilities: {spec_s:.3f} (95% CI: [{ci_lower_spec_s:.3f}, {ci_upper_spec_s:.3f}])")

""" SECTION: Calculate Predicted Proportion of Recombinants """
print()
print("SECTION: Calculate Predicted Proportion of Recombinants")
prop_breakpnt = (sum(cases_positive_index_breakpnt) + sum(controls_positive_index_breakpnt))/(n_cases + n_controls)
prop_s = (sum(cases_positive_index_s) + sum(controls_positive_index_s))/(n_cases + n_controls)

boot_prop_breakpnt = []
boot_prop_s = []
for i in range(len(boot_cases_breakpnt_list)):
	boot_prop_breakpnt.append((sum(boot_cases_breakpnt_list[i]) + sum(boot_controls_breakpnt_list[i]))/(n_cases + n_controls))
for i in range(len(boot_cases_s_list)):
	boot_prop_s.append((sum(boot_cases_s_list[i]) + sum(boot_controls_s_list[i]))/(n_cases + n_controls))

# Calculate bootstrap percentile intervals
ci_lower_prop_breakpnt, ci_upper_prop_breakpnt = np.percentile(boot_prop_breakpnt, [2.5, 97.5])
ci_lower_prop_s, ci_upper_prop_s = np.percentile(boot_prop_s, [2.5, 97.5])

print(f"Predicted proportion of recombinants using breakpoint: {prop_breakpnt:.3f} (95% CI: [{ci_lower_prop_breakpnt:.3f}, {ci_upper_prop_breakpnt:.3f}])")
print(f"Predicted proportion of recombinants using s: {prop_s:.3f} (95% CI: [{ci_lower_prop_s:.3f}, {ci_upper_prop_s:.3f}])")

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
overall_accuracy = np.mean(true_recombinant_sequences_flattened == inferred_recombinant_sequences_flattened)

boot_overall_accuracy = []
for _ in range(500):
	sampled_indices = np.random.choice(n_cases, size=n_cases, replace=True)  
	boot_overall_accuracy.append(np.mean(accuracy[sampled_indices]))

ci_lower_overall_accuracy, ci_upper_overall_accuracy = np.percentile(boot_overall_accuracy, [2.5, 97.5])

print(f"Position-by-position accuracy for all simulated recombinant sequences: {overall_accuracy:.3f} (95% CI: [{ci_lower_overall_accuracy:.3f}, {ci_upper_overall_accuracy:.3f}])")

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

breakpoint_distances = []
for i in range(n_cases):
	if len(inferred_breakpoints_list[i]) == len(breakpoints[i]):
		for j in range(len(inferred_breakpoints_list[i])):
			breakpoint_distances.append(np.abs(inferred_breakpoints_list[i][j] - breakpoints[i][j]))

print(f"Mean distance between true and inferred breakpoints: {round(np.mean(breakpoint_distances))}")

# Create histogram
plt.hist(breakpoint_distances, edgecolor='black')
# Add labels and title
plt.xlabel("Distance between true and inferred breakpoint")
plt.ylabel("Frequency")
# Show the plot
plt.tight_layout()
# Save figure as PNG
plt.savefig('output/results/breakpoint_distances.png', dpi=100, bbox_inches='tight')
# Clear the figure
plt.clf()

""" SECTION: Hamming Distance vs. Prediction Accuracy """
# print()
# print("SECTION: Hamming Distance vs. Prediction Accuracy")
# We save the sequences leading to the true recombinant here
sequence_set_list = []
# Filter the reference set for each tuple of chosen IDs
for id_pair in sampled_seq:
    sequence_set = []
    # Filter rows matching either ID in the tuple, and get the sequence
    sequence_set.extend(ref[ref['ID'] == id_pair[0]]['Trimmed'].tolist())
    sequence_set.extend(ref[ref['ID'] == id_pair[1]]['Trimmed'].tolist())
    sequence_set_list.append(sequence_set)

# Apply to each sublist
hamming_distances = np.array([hamming_distance(pair[0], pair[1]) for pair in sequence_set_list])

# Reshape x for sklearn (requires 2D input)
hamming_distances = hamming_distances.reshape(-1, 1)

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
print()
print("SECTION: Hamming Distance vs. Sensitivity")

# Fit logistic regression model
model_sens = LogisticRegression()
model_sens.fit(hamming_distances, cases_positive_index_s)

# Generate smooth predictions
x_smooth = np.linspace(min(hamming_distances), max(hamming_distances), 100).reshape(-1, 1)
y_smooth = model_sens.predict_proba(x_smooth)[:, 1] 

# Get the beta coefficients (intercept and slope)
b0_sens, b1_sens = model_sens.intercept_[0], model_sens.coef_[0][0]

# Create scatter plot with smaller, more transparent points
plt.scatter(hamming_distances, cases_positive_index_s, marker='o', color='blue', s=20, alpha=0.1)

# Bootstrap confidence intervals
n_bootstraps = 500
b1_sens_boot = []
y_pred_samples = np.zeros((n_bootstraps, len(x_smooth)))

for i in range(n_bootstraps):
    # Resample the data
    indices = np.random.choice(len(hamming_distances), len(hamming_distances), replace=True)
    hamming_boot = hamming_distances[indices]
    positive_boot = np.array(cases_positive_index_s)[indices]
    
    # Fit logistic regression to the resampled data
    model_sens_boot = LogisticRegression()
    model_sens_boot.fit(hamming_boot.reshape(-1, 1), positive_boot)

    # Store beta1 estimate
    b1_sens_boot.append(model_sens_boot.coef_[0][0])
    
    # Predict probabilities for the smooth x values
    y_pred_samples[i, :] = model_sens_boot.predict_proba(x_smooth)[:, 1]

# Compute the 95% confidence interval
y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# Plot logistic regression probability curve
plt.plot(x_smooth, y_smooth, color='red', label="Predicted probability from logistic regression fit")

# Plot confidence band
plt.fill_between(x_smooth.ravel(), y_lower, y_upper, color='red', alpha=0.2, label="95% CI")

# Add labels and title
plt.xlabel("Hamming distance between parental sequences")
plt.ylabel("Classified as a recombinant?")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)

# Show the plot
plt.tight_layout()

# Save figure as PNG
plt.savefig('output/results/sens_vs_hamming.png', dpi=100, bbox_inches='tight')

# Clear the figure
plt.clf()

# Compute the 95% confidence interval for beta1
b1_sens_lower = np.percentile(b1_sens_boot, 2.5)
b1_sens_upper = np.percentile(b1_sens_boot, 97.5)

print(f"Estimated logistic regression slope: {b1_sens:.3f} (95% CI: [{b1_sens_lower:.3f}, {b1_sens_upper:.3f}])")

""" SECTION: Plot First Few Recombinant Sequences and Inferred Lineages """
print()
print("SECTION: Plot First Few Recombinant Sequences and Inferred Lineage")
recombinant_sequences_filt = true_recombinant_sequences[:20]
inferred_recombinant_sequences_filt = inferred_recombinant_sequences[:20]

# Get unique lineage labels and map them to numeric values (same for both sets)
unique_lineages = sorted(set(lin for seq in list(recombinant_sequences_filt) + list(inferred_recombinant_sequences_filt) for lin in seq))
lineage_map = {lin: i for i, lin in enumerate(unique_lineages)}

# Convert sequences to numeric matrices
recombinant_sequences_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in recombinant_sequences_filt])
inferred_recombinant_sequences_matrix = np.array([[lineage_map[lin] for lin in seq] for seq in inferred_recombinant_sequences_filt])

# Choose a colormap (same for both sets)
num_colors = len(unique_lineages)
cmap = sns.color_palette("husl", num_colors)
cmap = plt.cm.colors.ListedColormap(cmap)

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(13, 10))  # 1 row, 2 columns

# Plot 1: Lineage Sequences Visualization for the first set
ax[0].imshow(recombinant_sequences_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
ax[0].set_xlabel("Position")
ax[0].set_ylabel("Sequence Index")
ax[0].set_title("True Lineages")

# Plot 2: Lineage Sequences Visualization for the second set
ax[1].imshow(inferred_recombinant_sequences_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Sequence Index")
ax[1].set_title("Inferred Lineages")

# Create a legend for both plots (same legend for both sets)
legend_patches = [mpatches.Patch(color=cmap(i), label=lineage) for lineage, i in lineage_map.items()]
fig.legend(handles=legend_patches, title="Lineage", bbox_to_anchor=(1.05, 0.5), loc='center left')
# Show the plot
plt.tight_layout()
# Save figure as PNG
plt.savefig('output/results/first_20_true_and_inferred.png', dpi=100, bbox_inches='tight')
# Clear the figure
plt.clf()

""" SECTION: Write Results to File """
print()
print("SECTION: Write Results to File")

with open("output/results/sim_results.txt", "w") as f:
    f.write(f"Number of cases: {round(n_cases)}\n")
    f.write(f"Number of controls: {round(n_controls)}\n")
    f.write(f"Length of trimmed genome: {round(N)}\n")
    f.write(f"Number of Pango lineages after collapsing: {round(M)}\n")
    f.write(f"Pango lineages after collapsing: {Pango_lineage_set}\n")

    f.write("\nSECTION: Calculate Sensitivity and Specificity\n")
    f.write(f"Sensitivity using breakpoint: {sens_breakpnt:.3f} (95% CI: [{ci_lower_sens_breakpnt:.3f}, {ci_upper_sens_breakpnt:.3f}])\n")
    f.write(f"Specificity using breakpoint: {spec_breakpnt:.3f} (95% CI: [{ci_lower_spec_breakpnt:.3f}, {ci_upper_spec_breakpnt:.3f}])\n")
    f.write(f"Sensitivity using estimated transition probabilities: {sens_s:.3f} (95% CI: [{ci_lower_sens_s:.3f}, {ci_upper_sens_s:.3f}])\n")
    f.write(f"Specificity using estimated transition probabilities: {spec_s:.3f} (95% CI: [{ci_lower_spec_s:.3f}, {ci_upper_spec_s:.3f}])\n")

    f.write("\nSECTION: Calculate Predicted Proportion of Recombinants\n")
    f.write(f"Predicted proportion of recombinants using breakpoint: {prop_breakpnt:.3f} (95% CI: [{ci_lower_prop_breakpnt:.3f}, {ci_upper_prop_breakpnt:.3f}])\n")
    f.write(f"Predicted proportion of recombinants using s: {prop_s:.3f} (95% CI: [{ci_lower_prop_s:.3f}, {ci_upper_prop_s:.3f}])\n")

    f.write("\nSECTION: Calculate Position-By-Position Accuracy\n")
    f.write(f"Position-by-position accuracy for all simulated recombinant sequences: {overall_accuracy:.3f} (95% CI: [{ci_lower_overall_accuracy:.3f}, {ci_upper_overall_accuracy:.3f}])\n")

    f.write("\nSECTION: Calculate Number of Matching Pango Lineages\n")
    f.write(f"Proportion of matching lineages for recombinants: {match_prob:.3f} (95% CI: [{ci_lower_match_prob:.3f}, {ci_upper_match_prob:.3f}])\n")
    f.write(f"Proportion of partially matching lineages for recombinants: {partial_match_prob:.3f} (95% CI: [{ci_lower_partial_match_prob:.3f}, {ci_upper_partial_match_prob:.3f}])\n")
    f.write(f"Proportion of matching lineages for controls: {match_prob_controls:.3f} (95% CI: [{ci_lower_match_prob_controls:.3f}, {ci_upper_match_prob_controls:.3f}])\n")
    f.write(f"Proportion of partially matching lineages for controls: {partial_match_prob_controls:.3f} (95% CI: [{ci_lower_partial_match_prob_controls:.3f}, {ci_upper_partial_match_prob_controls:.3f}])\n")

    f.write("\nSECTION: Calculate Breakpoint Distances\n")
    f.write(f"Mean distance between true and inferred breakpoints: {round(np.mean(breakpoint_distances))}\n")

    f.write("\nSECTION: Hamming Distance vs. Sensitivity\n")
    f.write(f"Estimated logistic regression slope: {b1_sens:.3f} (95% CI: [{b1_sens_lower:.3f}, {b1_sens_upper:.3f}])\n")
