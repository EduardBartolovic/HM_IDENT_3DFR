import re
import numpy as np
from collections import defaultdict

# Load log text
with open("E:\\Download\\performance_perspective.txt", "r") as f:
    log = f.read()


def fmt(v): return f"{v:+.3f}" if not np.isnan(v) else "N/A"#++++++++++++++++++++++++++++++++++++++++++++++++
#def fmt(v): return f"{v:+.2f}" if not np.isnan(v) else "N/A"

# --- STEP 1: Split by backbone blocks ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)

# Regex pattern to match each dataset evaluation
eval_pattern = re.compile(
    r"(rgb_bff_crop8(?:_swap_ref\d+)?)\s+\S+:"
    r".*?Front RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
    r"Concat RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
    r"Concat_Mean RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
    r"Concat_PCA RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
    r"Score_prod MRR:\s*([\d\.NA]+).*?"
    r"Score_mean MRR:\s*([\d\.NA]+).*?"
    r"Score_max MRR:\s*([\d\.NA]+).*?"
    r"Score_maj MRR:\s*([\d\.NA]+).*?"
    r"MV RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+)",
    flags=re.S
)

# --- STEP 2: Gather per-dataset results across backbones ---
datasets = defaultdict(lambda: defaultdict(list))

for block in backbone_blocks[1:]:
    evals = eval_pattern.findall(block)
    for ev in evals:
        dataset = ev[0]
        vals = {
            "Front": ev[2], "Concat": ev[4], "Concat_Mean": ev[6],
            "Score_Prod": ev[7], "Score_Mean": ev[8], "Score_Max": ev[9],
            "Score_Maj": ev[10], "MV": ev[12]
        }
        for k, v in vals.items():
            if v != "N/A":
                datasets[dataset][k].append(float(v))

# --- STEP 3: Average across backbones ---
avg_results = {}
for dataset, methods in datasets.items():
    avg_results[dataset] = {k: np.nanmean(v) for k, v in methods.items()}

# --- STEP 4: Sort datasets by number in their name (swap_refXX order) ---
def dataset_key(d):
    match = re.search(r"swap_ref(\d+)", d)
    return int(match.group(1)) if match else 0

avg_results = dict(sorted(avg_results.items(), key=lambda x: dataset_key(x[0])))

# --- STEP 5: LaTeX table ---
print("\\begin{table*}[t]")
print("\\centering")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Dataset} & \\textbf{Front} & \\multicolumn{3}{c|}{\\textbf{Concat}} & "
      "\\multicolumn{3}{c|}{\\textbf{Score}} & \\textbf{Feature-Agg} \\\\")
print("\\cline{3-9}")
print("& \\textbf{Only} & \\textbf{Full $\\downarrow$} & \\textbf{Mean} & "
      "\\textbf{Prod $\\downarrow$} & \\textbf{Mean $\\downarrow$} & \\textbf{Max $\\downarrow$} & "
      "\\textbf{Maj $\\downarrow$} & \\textbf{M5} \\\\")
print("\\hline")

# Mapping from dataset name to readable label (e.g. BFF₀.₀–₀)
ref_to_label = {
    0: "BFF$_{0}$",
    9: "BFF$_{9}$",
    11: "BFF$_{11}$",
    14: "BFF$_{14}$",
    15: "BFF$_{15}$",
    19: "BFF$_{19}$",
    23: "BFF$_{23}$",
    27: "BFF$_{27}$",
    32: "BFF$_{32}$"
}

for dataset, vals in avg_results.items():
    # Extract numeric order for naming
    ref = re.search(r"swap_ref(\d+)", dataset)
    refnum = int(ref.group(1)) if ref else 0
    name = ref_to_label.get(refnum, dataset)

    # Prepare row values
    row = {
        "Front": vals.get("Front", np.nan),
        "Concat": vals.get("Concat", np.nan),
        "Concat_Mean": vals.get("Concat_Mean", np.nan),
        "Score_Prod": vals.get("Score_Prod", np.nan),
        "Score_Mean": vals.get("Score_Mean", np.nan),
        "Score_Max": vals.get("Score_Max", np.nan),
        "Score_Maj": vals.get("Score_Maj", np.nan),
        "MV": vals.get("MV", np.nan),
    }

    # Identify best (blue) and worst (orange)
    numeric_vals = {k: v for k, v in row.items() if not np.isnan(v) and k != "Front"}
    best_k = max(numeric_vals, key=numeric_vals.get)
    worst_k = min(numeric_vals, key=numeric_vals.get)

    def highlight(k):
        v = row[k]
        if np.isnan(v):
            return "N/A"
        val = fmt(v)
        if k == best_k:
            return f"\\cellcolor{{blue!15}}{val}"
        elif k == worst_k:
            return f"\\cellcolor{{orange!20}}{val}"
        return val

    print(f"{name} & {fmt(row['Front'])} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Score_Prod')} & {highlight('Score_Mean')} & "
          f"{highlight('Score_Max')} & {highlight('Score_Maj')} & {highlight('MV')}\\\\")
    print("\\hline")

print("\\end{tabular}")
print("\\caption{Averaged MRR results over 8 views on the BFF dataset, showing performance across increasing levels of perspective error for all models. "
      "An increase in perspective error results in strong performance drops in most methods. Blue marks the best, orange the worst.}")
print("\\label{tab:perspective_error}")
print("\\end{table*}")