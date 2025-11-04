import re
import numpy as np

# Load log text
with open("E:\\Download\\performace.txt", "r") as f:
    log = f.read()

def fmt(v):
    return f"{v:+.2f}" if not np.isnan(v) else "N/A"

# --- Parse log into backbone blocks ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)
results = []

for block in backbone_blocks[1:]:
    # Extract backbone name
    backbone_match = re.search(r"[\\/](?:[^\\/]+)\.(pt|pth)", block)
    backbone = re.search(r"([^\\/]+)\.(?:pt|pth)", backbone_match.group(0)).group(1).replace("_", "-") if backbone_match else "Unknown"

    # Extract multipie_crop8 block
    dataset_blocks = re.findall(r"Perform.*?multipie_crop8.*", block, flags=re.S)
    if not dataset_blocks:
        continue
    dataset_block = dataset_blocks[0]

    methods = ["Front", "Concat", "Concat_Mean", "Concat_PCA",
               "Score_sum", "Score_prod", "Score_mean", "Score_max", "MV"]
    all_vals = {}

    for m in methods:
        # Search for GBIG for each method separately
        match = re.search(rf"{m}.*?GBIG:\s*([\d\.]+)", dataset_block)
        if match:
            all_vals[m] = float(match.group(1))

    if "Front" not in all_vals:
        continue  # Skip if no Front value

    base = all_vals["Front"]
    gains = {k: all_vals[k]-base for k in all_vals.keys() if k != "Front"}

    results.append((backbone, all_vals, gains))

# --- Generate LaTeX table ---
print("\\begin{table*}[t]")
print("\\centering")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\multicolumn{3}{c|}{\\textbf{Concat}} & "
      "\\multicolumn{3}{c|}{\\textbf{Score}} & \\textbf{Feature-Agg} \\\\")
print("\\cline{3-9}")
print("& \\textbf{Only} & \\textbf{Full} & \\textbf{Mean} & \\textbf{PCA} & "
      "\\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\textbf{MV} \\\\")
print("\\hline")

for backbone, avg, gains in results:
    # Highlight best gain
    valid_gains = {k: v for k, v in gains.items() if not np.isnan(v)}
    best_key = max(valid_gains, key=valid_gains.get) if valid_gains else None

    def highlight(k):
        if k not in gains:
            return "N/A"
        v = gains[k]
        val = fmt(v)
        return f"\\cellcolor{{blue!15}}{{{val}}}" if k == best_key else val

    print(f"{backbone} & {avg['Front']:.3f} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Concat_PCA')} & {highlight('Score_prod')} & {highlight('Score_mean')} & "
          f"{highlight('Score_max')} & {highlight('MV')}\\\\")
    print("\\hline")

print("\\end{tabular}")
print("\\caption{Absolute GBIG gains for the Multipie dataset compared to single-front view.}")
print("\\label{tab:gbig_multipie}")
print("\\end{table*}")
