import pandas as pd
import glob
import os
import gzip
import json
from collections import Counter
from Bio import SeqIO
import zstandard as zstd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import plotly.express as px
import webbrowser
import tempfile

from entropy_utils import summarize_entropy_from_multiple_json, plot_lineage_pair_paths

# ------------------------------
# Load Expected Recombinant Counts
# ------------------------------
# def load_expected_counts(pattern, frequency_file,
#                          alpha=1, k=1,
#                          N_t=1000,            # ← constant sample size
#                          scale=1):
#     """
#     Expected = 2·pA·pB · N_t · prevalence^alpha · k · scale
#               (with N_t fixed at 1 000 for every window)
#     """
#     # prevalence table
#     freq_df = pd.read_csv(
#         frequency_file, sep='\t', parse_dates=["start", "end"]
#     )

#     dfs = []
#     for fp in sorted(glob.glob(pattern)):
#         # ---- extract window dates from filename ----
#         start_str, end_str = os.path.basename(fp)\
#                                .replace("expected_recombinants_", "")\
#                                .replace(".csv.gz", "")\
#                                .split("_")
#         start, end = pd.to_datetime(start_str), pd.to_datetime(end_str)

#         # ---- prevalence for this window ----
#         freq_row = freq_df.query("start == @start and end == @end")
#         if freq_row.empty:
#             print(f"⚠️ prevalence missing for {start_str}_{end_str}")
#             continue
#         prevalence = freq_row.iloc[0]["mean_frequency"]

#         # ---- read pairwise 2·pA·pB frequencies ----
#         df = pd.read_csv(fp)

#         df["Expected_Count"] = (
#             df["Expected_Recombinant_Frequency"] *
#             N_t *
#             (prevalence ** alpha) *
#             k *
#             scale
#         )

#         # canonicalise lineage order
#         df[["Lineage_1", "Lineage_2"]] = (
#             df[["Lineage_A", "Lineage_B"]]
#             .apply(sorted, axis=1, result_type="expand")
#         )
#         dfs.append(df)

#     combined = pd.concat(dfs, ignore_index=True)

#     return (combined
#             .groupby(["Lineage_1", "Lineage_2"], as_index=False)
#             ["Expected_Count"].sum())

# # ------------------------------
# # Load Inferred Recombinant Counts
# # ------------------------------
# def load_inferred_counts(pattern):
#     files = glob.glob(pattern)
#     counter = Counter()
#     for fp in files:
#         with gzip.open(fp, 'rt') as f:
#             data = json.load(f)
#         for segments in data.values():
#             parents = set()
#             for seg in segments:
#                 if isinstance(seg, list) and len(seg) == 3:
#                     parents.update([seg[0], seg[2]])
#             if len(parents)==2:
#                 pair = tuple(sorted(parents))
#                 counter[pair] += 1
#     return pd.DataFrame([{"Lineage_1":a,"Lineage_2":b,"Inferred_Count":c} for (a,b),c in counter.items()])

# # ------------------------------
# # Consensus + Hamming Distance
# # ------------------------------
# def load_consensus(fasta_zst):
#     seqs={}
#     with open(fasta_zst,'rb') as fh:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(fh) as reader:
#             stream = io.TextIOWrapper(reader, encoding='utf-8')
#             for rec in SeqIO.parse(stream,'fasta'):
#                 seqs[rec.id] = str(rec.seq).upper()
#     return seqs

# def hamming(s1, s2):
#     assert len(s1) == len(s2)
#     diffs = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b and a != 'N' and b != 'N']
#     return len(diffs), diffs

# def hamming_with_positions(row):
#     if row["Lineage_1"] in cons and row["Lineage_2"] in cons:
#         dist, pos = hamming(cons[row["Lineage_1"]], cons[row["Lineage_2"]])
#         return pd.Series({"Hamming_Distance": dist, "Mutation_Positions": pos})
#     return pd.Series({"Hamming_Distance": None, "Mutation_Positions": None})

# # ------------------------------
# # RUN PIPELINE
# # ------------------------------
# exp = load_expected_counts(
#     "../run-on-cluster-do-not-edit/real-data-analysis/output/sliding_windows/expected_recombinant_freq/expected_recombinants_*.csv.gz",
#     "../data/mean_frequency_by_test_window.tsv"
# )
# inf = load_inferred_counts("../run-on-cluster-do-not-edit/real-data-analysis/output/sliding_windows/inferred/inferred_lineages_*.json.gz")
# final = pd.merge(exp, inf, on=["Lineage_1","Lineage_2"], how="outer").fillna(0)
# final["Inferred_Count"] = final["Inferred_Count"].astype(int)

# # Hamming
# cons = load_consensus("../data/pango-consensus-sequences_genome-nuc.fasta.zst")
# final[["Hamming_Distance", "Mutation_Positions"]] = final.apply(hamming_with_positions, axis=1)

# # Entropy summaries
# entropy_df = summarize_entropy_from_multiple_json("../run-on-cluster-do-not-edit/real-data-analysis/output/sliding_windows/inferred/inferred_lineages_*.json.gz")
# final = final.merge(entropy_df, on=["Lineage_1","Lineage_2"], how="left")

# # Get frequency of inferred
# final["Prop_Inferred"] = final["Inferred_Count"] / final["Expected_Count"]

# final = final[final["Expected_Count"] > 0]
# # final = final[final["Inferred_Count"] >= 20]

# # Get top 20 lineage combinations with highest frequency of inferred lineages
# top20 = final.sort_values(by="Prop_Inferred", ascending=False).head(20)
# print(top20.drop(["Max_Entropy", "Positions_Over_0.5", "Mutation_Positions"], axis=1))

# # Save cleaned data
# final.to_csv("recombinant_counts_summary.csv", index=False)

# plt.figure(figsize=(8, 6))
# ax = plt.gca()

# scatter = sns.scatterplot(
#     data=final,
#     x="Hamming_Distance",
#     y="Prop_Inferred",
#     hue="Mean_Entropy",
#     size="Inferred_Count",
#     sizes=(20, 200),  # You can adjust min/max size
#     palette="viridis",
#     alpha=0.8,
#     ax=ax
# )

# # Remove legend (optional, to avoid overlapping)
# ax.legend_.remove()

# # Manual colorbar for Mean Entropy
# norm = mcolors.Normalize(vmin=final["Mean_Entropy"].min(), vmax=final["Mean_Entropy"].max())
# sm = cm.ScalarMappable(cmap="viridis", norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label("Mean Entropy")

# # Labels and title
# plt.xlabel("Hamming Distance")
# plt.ylabel("Proportion Inferred")
# plt.title("Proportion of Recombinants Inferred vs Hamming Distance\nColored by Entropy, Sized by Inferred Count")

# plt.tight_layout()
# plt.savefig("prop_inferred_vs_hamming_entropy_size.png", dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(8, 6))
# ax = plt.gca()

# Scatter: colored by Hamming Distance, uniform size

# ── Optional: log-transform the distance for the colour axis ────────────────
# ── Pre-process: log-scaled colour field ──────────────────────────────────


final = pd.read_csv("recombinant_counts_summary.csv") 
final["HD1"] = final["Hamming_Distance"].clip(lower=1)  # avoid log10(0)
final["log_HD"] = np.log10(final["HD1"])

# ── Build interactive figure (no size channel) ───────────────────────────
fig = px.scatter(
    final,
    x="Expected_Count",
    y="Inferred_Count",
    color="log_HD",                      # colour = log10(Hamming distance)
    color_continuous_scale="Viridis",
    hover_data={
        "Lineage_1": True,
        "Lineage_2": True,
        "Hamming_Distance": True,
        "Expected_Count": ":.1f",
        "Inferred_Count": ":.1f",
        "log_HD": False                 # hide helper field in tooltip
    },
    labels={
        "Expected_Count": "Expected Count  (k = 1, α = 1, N = 1000)",
        "Inferred_Count": "Inferred Count",
        "log_HD": "Hamming Distance (log₁₀)"
    },
    title="Inferred vs Expected Recombinant Counts<br>"
          "Colour = Hamming Distance (log scale)"
)

# ── Add y = x guide line ─────────────────────────────────────────────────
lims = [min(fig.data[0].x.min(), fig.data[0].y.min()),
        max(fig.data[0].x.max(), fig.data[0].y.max())]

fig.add_shape(
    type="line",
    x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
    line=dict(color="gray", dash="dash"), layer="below"
)

fig.update_xaxes(range=[lims[0], lims[1]])
fig.update_yaxes(range=[lims[0], lims[1]])

fig.update_coloraxes(
    colorbar_title="log₁₀ (Hamming Distance)",
    cmin=final["log_HD"].min(),
    cmax=final["log_HD"].max()
)

# ── Save and auto-open in browser ────────────────────────────────────────
fig.write_html("inferred_vs_expected_interactive.html", auto_open=True)
#############

lineage_pairs_of_interest = {
    tuple(sorted(["AY.5", "B.1.617.2"])),
    tuple(sorted(["AY.9", "B.1.617.2"])),
    tuple(sorted(["B.1.1.7", "B.1.177"])),
    tuple(sorted(["BA.1.1", "BA.2"])),
    tuple(sorted(["BA.1", "BA.2"])),
    tuple(sorted(["BE.1.1", "BQ.1.1"]))    
}

# Filter to just the pairs of interest
final = final[final[["Lineage_1", "Lineage_2"]].apply(lambda row: tuple(sorted(row)) in lineage_pairs_of_interest, axis=1)]

# === Plot Recombinant Paths for High Inferred Cases ===
json_pattern = "../run-on-cluster-do-not-edit/real-data-analysis/output/sliding_windows/inferred/inferred_lineages_*.json.gz"

for _, row in final.iterrows():
    lineage_1 = row["Lineage_1"]
    lineage_2 = row["Lineage_2"]
    mean_entropy = row.get("Mean_Entropy", float('nan'))
    max_entropy = row.get("Max_Entropy", float('nan'))
    pos_over_0_5 = row.get("Positions_Over_0.5", float('nan'))
    hamming = row.get("Hamming_Distance", float('nan'))
    mutation_pos = row.get("Mutation_Positions", [])

    caption = (
        f"Mean entropy: {mean_entropy:.3f}, "
        f"Max entropy: {max_entropy:.3f}, "
        f"# positions > 0.5: {int(pos_over_0_5)}, "
        f"Hamming distance: {hamming:.1f}"
    )

    plot_lineage_pair_paths(
        json_pattern=json_pattern,
        lineage_pair=(lineage_1, lineage_2),
        output_dir="figs",
        caption=caption,
        mutation_positions=mutation_pos
    )
