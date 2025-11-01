import re
import matplotlib.pyplot as plt
import collections
import numpy as np

# TODO Add perspective Error for every dataset on x

# Load your text file
with open("E:\\Download\\performance_perspective.txt", "r") as f:
    log = f.read()

# Organize data per model
model_data = collections.defaultdict(lambda: {"x": [], "Front": [], "Concat": [], "Concat_Mean": [], "Concat_PCA": [], "Score_Prod": [], "Score_Mean": [], "Score_Max": [], "Score_Maj": [], "MV": []})

current_model = None
for line in log.splitlines():
    # Detect model
    m = re.search(r"Loading ONLY Backbone Checkpoint (.+?)\.pt", line)
    if m:
        current_model = m.group(1).split("/")[-1]
        continue

    # Extract dataset and all MRR metrics
    dm = re.search(
        r"(rgb_bff_crop8(?:_swap_ref\d{2})?).*?"
        r"Front RR1: [0-9.]+ MRR: ([0-9.]+).*?"
        r"Concat RR1: [0-9.]+ MRR: ([0-9.]+).*?"
        r"Concat_Mean RR1: [0-9.]+ MRR: ([0-9.]+).*?"
        r"Concat_PCA RR1: [0-9.]+ MRR: ([0-9.]+).*?"
        r"Score_prod MRR: ([0-9.]+).*?"
        r"Score_mean MRR: ([0-9.]+).*?"
        r"Score_max MRR: ([0-9.]+).*?"
        r"Score_maj MRR: ([0-9.]+).*?"
        r"MV RR1: [0-9.]+ MRR: ([0-9.]+)", line
    )
    if dm and current_model:
        dataset_name = dm.group(1)
        x_val = int(dataset_name[-2:]) if "swap_ref" in dataset_name else 0

        model_data[current_model]["x"].append(x_val)
        model_data[current_model]["Front"].append(float(dm.group(2)))
        model_data[current_model]["Concat"].append(float(dm.group(3)))
        model_data[current_model]["Concat_Mean"].append(float(dm.group(4)))
        model_data[current_model]["Concat_PCA"].append(float(dm.group(5)))
        model_data[current_model]["Score_Prod"].append(float(dm.group(6)))
        model_data[current_model]["Score_Mean"].append(float(dm.group(7)))
        model_data[current_model]["Score_Max"].append(float(dm.group(8)))
        model_data[current_model]["Score_Maj"].append(float(dm.group(9)))
        model_data[current_model]["MV"].append(float(dm.group(10)))

# Normalize x-axis to 0-1
for values in model_data.values():
    max_x = max(values["x"])
    values["x"] = [x/max_x for x in values["x"]]

# --- Compute averages ---
# Assume all models have same x positions
x_positions = sorted(list(model_data[next(iter(model_data))]["x"]))
avg_metrics = {k: [] for k in ["Front","Concat","Concat_Mean","Concat_PCA","Score_Prod","Score_Mean","Score_Max","Score_Maj","MV"]}

for i in range(len(x_positions)):
    for key in avg_metrics.keys():
        vals = [model[key][i] for model in model_data.values()]
        avg_metrics[key].append(np.mean(vals))

# Plot (just some main curves as example)
plt.figure(figsize=(12,6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

for i, (model_name, values) in enumerate(model_data.items()):
    color = colors[i % len(colors)]
    plt.plot(values["x"], values["Front"], marker='o', linestyle='-', color=color, alpha=0.5, label=f'{model_name} Front MRR')
    plt.plot(values["x"], values["Concat"], marker='s', linestyle='--', color=color, alpha=0.5, label=f'{model_name} Concat MRR')
    plt.plot(values["x"], values["MV"], marker='^', linestyle=':', color=color, alpha=0.5, label=f'{model_name} MV MRR')

# Plot averaged curves
for key, style, marker in zip(["Front","Concat","MV"], ['-','--',':'], ['o','s','^']):
    plt.plot(x_positions, avg_metrics[key], color='black', linestyle=style, marker=marker, linewidth=2.5, label=f'Average {key} MRR')

plt.xticks([i/10 for i in range(0, 11)])
plt.xlabel("Randomness (0 → 1)")
plt.ylabel("Accuracy MRR (%)")
plt.ylim(96, 100)
plt.title("MRR over increasing randomness for all models")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# --- Print LaTeX table with averaged results ---
print("\n" + "=" * 80)
print("LATEX TABLE (AVERAGED RESULTS)")
print("=" * 80 + "\n")

print("\\begin{table*}[t]")
print("\\centering")
print("\\label{tab:perspective_error}")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Concat} & \\textbf{Concat-Mean} & \\textbf{Concat-PCA} & \\textbf{Score-Prod} & \\textbf{Score-Mean} & \\textbf{Score-Max} & \\textbf{Score-Maj} & \\textbf{Feature-Agg}\\\\")
print("\\hline")

for i, x in enumerate(x_positions):
    label = f"BFF\\textsubscript{{{x:.1f}}}"
    front = f"{avg_metrics['Front'][i]:.2f}"
    concat = f"{avg_metrics['Concat'][i]:.2f}"
    concat_mean = f"{avg_metrics['Concat_Mean'][i]:.2f}"
    concat_pca = f"{avg_metrics['Concat_PCA'][i]:.2f}"
    score_prod = f"{avg_metrics['Score_Prod'][i]:.2f}"
    score_mean = f"{avg_metrics['Score_Mean'][i]:.2f}"
    score_max = f"{avg_metrics['Score_Max'][i]:.2f}"
    score_maj = f"{avg_metrics['Score_Maj'][i]:.2f}"
    mv = f"{avg_metrics['MV'][i]:.2f}"

    print(f"{label} & {front} & {concat} & {concat_mean} & {concat_pca} & {score_prod} & {score_mean} & {score_max} & {score_maj} & {mv}\\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{MRR results with 8 views on the Vox2train dataset, averaged across all models.}")
print("\\end{table*}")