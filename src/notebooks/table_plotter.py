import re
import numpy as np
from collections import defaultdict

# Load log text
with open("E:\\Download\\performace.txt", "r") as f:
    log = f.read()

# --- STEP 1: Parse log into backbone blocks ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)
results = []

for block in backbone_blocks[1:]:
    # Extract backbone name (cleaned up)
    backbone_match = re.search(r"[\\/](?:[^\\/]+)\.(pt|pth)", block)
    if backbone_match:
        # Extract just the filename without extension
        backbone = re.search(r"([^\\/]+)\.(?:pt|pth)", backbone_match.group(0)).group(1).replace("_", "-")
    else:
        backbone = "Unknown"

    # Extract MRR values per test set
    evals = re.findall(
        r"(\b(?:vox2train_crop8)\b).*?Front RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?" # vox2train_crop8  rgb_bff_crop8 ytf_crop8 ytf_crop8
        r"Concat RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
        r"Concat_Mean RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
        r"Concat_PCA RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
        r"Score_prod MRR:\s*([\d\.NA]+).*?"
        r"Score_mean MRR:\s*([\d\.NA]+).*?"
        r"Score_max MRR:\s*([\d\.NA]+).*?"
        r"Score_maj MRR:\s*([\d\.NA]+).*?"
        r"MV RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+)",
        block, flags=re.S,
    )

    if not evals:
        continue

    # Average MRRs across datasets (BFF + Vox2Test)
    all_vals = defaultdict(list)
    for ev in evals:
        dataset = ev[0]
        all_vals["Front"].append(float(ev[2]) if ev[2] != "N/A" else np.nan)
        all_vals["Concat"].append(float(ev[4]) if ev[4] != "N/A" else np.nan)
        all_vals["Concat_Mean"].append(float(ev[6]) if ev[6] != "N/A" else np.nan)
        all_vals["Concat_PCA"].append(float(ev[8]) if ev[8] != "N/A" else np.nan)
        all_vals["Score_Prod"].append(float(ev[9]) if ev[9] != "N/A" else np.nan)
        all_vals["Score_Mean"].append(float(ev[10]) if ev[10] != "N/A" else np.nan)
        all_vals["Score_Max"].append(float(ev[11]) if ev[11] != "N/A" else np.nan)
        all_vals["Score_Maj"].append(float(ev[12]) if ev[12] != "N/A" else np.nan)
        all_vals["MV"].append(float(ev[14]) if ev[13] != "N/A" else np.nan)

    # Compute averages
    avg = {k: np.nanmean(v) for k, v in all_vals.items()}

    # Compute relative gains (vs. Front)
    base = avg["Front"]
    gains = {k: avg[k] - base for k in avg.keys() if k != "Front"}

    results.append((backbone, avg, gains))

# --- STEP 2: Generate LaTeX table ---
print("\\begin{table*}[t]")
print("\\centering")
print("\\label{tab:3d_bff_gain}")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\multicolumn{3}{c|}{\\textbf{Concat}} & "
      "\\multicolumn{4}{c|}{\\textbf{Score}} & \\textbf{Feature-Agg} \\\\")
print("\\cline{3-10}")
print("& \\textbf{Only} & \\textbf{Full} & \\textbf{Mean} & \\textbf{PCA} & "
      "\\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\textbf{Maj} & \\textbf{M5} \\\\")
print("\\hline")

avg_fronts, avg_gains = [], defaultdict(list)

for backbone, avg, gains in results:
    def fmt(v): return f"{v:+.3f}" if not np.isnan(v) else "N/A"
    base = avg["Front"]
    avg_fronts.append(base)

    for k in gains:
        avg_gains[k].append(gains[k])

    # Find which gain is the best (highest)
    valid_gains = {k: v for k, v in gains.items() if not np.isnan(v)}
    best_key = max(valid_gains, key=valid_gains.get) if valid_gains else None

    def highlight(k):
        v = gains[k]
        if np.isnan(v):
            return "N/A"
        val = fmt(v)
        if k == best_key:
            return f"\\cellcolor{{blue!15}}{{{val}}}"
        return val

    print(f"{backbone} & {base:.3f} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Concat_PCA')} & {highlight('Score_Prod')} & {highlight('Score_Mean')} & "
          f"{highlight('Score_Max')} & {highlight('Score_Maj')} & {highlight('MV')}\\\\")
    print("\\hline")

# --- Average row ---
print("\\hline")
print("\\textbf{Average} & "
      f"{np.nanmean(avg_fronts):.3f} & "
      f"{fmt(np.nanmean(avg_gains['Concat']))} & "
      f"{fmt(np.nanmean(avg_gains['Concat_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Concat_PCA']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Prod']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Max']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Maj']))} & "
      f"{fmt(np.nanmean(avg_gains['MV']))} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Relative MRR gains (\\%) of multi-view fusion strategies compared to single-front view "
      "on the 3D-BFF dataset.}")
print("\\end{table*}")
