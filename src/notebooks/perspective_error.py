import re
import matplotlib.pyplot as plt
import collections
import numpy as np

# TODO Add perspective Error for every dataset on x

# Load your text file
with open("E:\\Download\\performance_perspective.txt", "r") as f:
    log = f.read()

# Organize data per model
model_data = collections.defaultdict(lambda: {"x": [], "Front": [], "Concat": [], "MV": []})

current_model = None
for line in log.splitlines():
    m = re.search(r"Loading ONLY Backbone Checkpoint (.+?)\.pt", line)
    if m:
        current_model = m.group(1).split("/")[-1]  # just filename
        continue
    dm = re.search(r"(rgb_bff_crop8(?:_swap_ref\d{2})?) .*?Front RR1: ([0-9.]+).*?Concat RR1: ([0-9.]+).*?MV RR1: ([0-9.]+)", line)
    if dm and current_model:
        dataset_name = dm.group(1)
        # Extract the last number in the dataset name
        if "swap_ref" in dataset_name:
            x_val = int(dataset_name[-2:])  # last two digits
        else:
            x_val = 0
        model_data[current_model]["x"].append(x_val)
        model_data[current_model]["Front"].append(float(dm.group(2)))
        model_data[current_model]["Concat"].append(float(dm.group(3)))
        model_data[current_model]["MV"].append(float(dm.group(4)))

# Normalize x-axis to 0-1
for values in model_data.values():
    max_x = max(values["x"])
    values["x"] = [x/max_x for x in values["x"]]

# --- Compute averages ---
# Assume all models have same x positions
x_positions = sorted(list(model_data[next(iter(model_data))]["x"]))
avg_front, avg_concat, avg_mv = [], [], []

for i in range(len(x_positions)):
    front_vals, concat_vals, mv_vals = [], [], []
    for model in model_data.values():
        front_vals.append(model["Front"][i])
        concat_vals.append(model["Concat"][i])
        mv_vals.append(model["MV"][i])
    avg_front.append(np.mean(front_vals))
    avg_concat.append(np.mean(concat_vals))
    avg_mv.append(np.mean(mv_vals))

# --- Plot ---
plt.figure(figsize=(12,6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

for i, (model_name, values) in enumerate(model_data.items()):
    color = colors[i % len(colors)]
    plt.plot(values["x"], values["Front"], marker='o', linestyle='-', color=color, alpha=0.5, label=f'{model_name} Front')
    plt.plot(values["x"], values["Concat"], marker='s', linestyle='--', color=color, alpha=0.5, label=f'{model_name} Concat')
    plt.plot(values["x"], values["MV"], marker='^', linestyle=':', color=color, alpha=0.5, label=f'{model_name} MV')

# Add average curves (bold + black)
plt.plot(x_positions, avg_front, color='black', linewidth=2.5, linestyle='-', marker='o', label='Average Front')
plt.plot(x_positions, avg_concat, color='black', linewidth=2.5, linestyle='--', marker='s', label='Average Concat')
plt.plot(x_positions, avg_mv, color='black', linewidth=2.5, linestyle=':', marker='^', label='Average MV')

plt.xticks([i/10 for i in range(0, 11)])
plt.xlabel("Randomness (0 → 1)")
plt.ylabel("Accuracy RR1 (%)")
plt.ylim(99.1, 100.01)
plt.title("Accuracy over increasing randomness for all models")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()