import re
import numpy as np
from collections import defaultdict

# Load log text
with open("E:\\Download\\performance_perspective.txt", "r") as f:
    log = f.read()


def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "N/A"


# --- STEP 1: Split into backbone blocks ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)

# --- STEP 2: Define regex pattern ---
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

# --- STEP 3: Parse data ---
all_backbones = {}
datasets = defaultdict(lambda: defaultdict(list))

for block in backbone_blocks[1:]:
    # Extract backbone name
    name_match = re.search(r"[\\/](?:[^\\/]+)\.(?:pt|pth)", block)
    backbone = re.search(r"([^\\/]+)\.(?:pt|pth)", name_match.group(0)).group(1) if name_match else "Unknown"

    evals = eval_pattern.findall(block)
    per_backbone = {}

    for ev in evals:
        dataset = ev[0]
        vals = {
            "Front": ev[2], "Concat": ev[4], "Concat_Mean": ev[6],
            "Score_Prod": ev[7], "Score_Mean": ev[8],
            "Score_Max": ev[9], "Score_Maj": ev[10], "MV": ev[12]
        }

        per_backbone[dataset] = {}
        for k, v in vals.items():
            if v != "N/A":
                per_backbone[dataset][k] = float(v)
                datasets[dataset][k].append(float(v))

    all_backbones[backbone] = per_backbone

# --- STEP 4: Average results across backbones ---
avg_results = {
    d: {k: np.nanmean(v) for k, v in m.items()}
    for d, m in datasets.items()
}


# --- Helper: sort datasets by swap_ref order ---
def dataset_key(d):
    match = re.search(r"swap_ref(\d+)", d)
    return int(match.group(1)) if match else 0


# --- Label mapping ---
ref_to_label = {
    0: "BFF$_{0}$", 3: "BFF$_{9}$", 4: "BFF$_{11}$", 5: "BFF$_{14}$",
    6: "BFF$_{15}$", 7: "BFF$_{19}$", 8: "BFF$_{23}$",
    9: "BFF$_{27}$", 10: "BFF$_{32}$"
}


# --- STEP 5: Print average table ---
def print_table(title, data):
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Dataset} & \\textbf{Front} & \\multicolumn{2}{c|}{\\textbf{Concat}} & "
          "\\multicolumn{4}{c|}{\\textbf{Score}} \\\\")
    print("\\cline{3-8}")
    print("& \\textbf{Only} & \\textbf{Full} & \\textbf{Mean} & "
          "\\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\textbf{Maj} \\\\")
    print("\\hline")

    for dataset, vals in sorted(data.items(), key=lambda x: dataset_key(x[0])):
        ref = re.search(r"swap_ref(\d+)", dataset)
        refnum = int(ref.group(1)) if ref else 0
        name = ref_to_label.get(refnum, dataset)

        row = {
            "Front": vals.get("Front", np.nan),
            "Concat": vals.get("Concat", np.nan),
            "Concat_Mean": vals.get("Concat_Mean", np.nan),
            "Score_Prod": vals.get("Score_Prod", np.nan),
            "Score_Mean": vals.get("Score_Mean", np.nan),
            "Score_Max": vals.get("Score_Max", np.nan),
            "Score_Maj": vals.get("Score_Maj", np.nan),
        }

        numeric_vals = {k: v for k, v in row.items() if not np.isnan(v) and k != "Front"}
        best_k = max(numeric_vals, key=numeric_vals.get)
        worst_k = min(numeric_vals, key=numeric_vals.get)

        def highlight(k):
            v = row[k]
            if np.isnan(v): return "N/A"
            val = fmt(v)
            if k == best_k:
                return f"\\cellcolor{{blue!15}}{val}"
            elif k == worst_k:
                return f"\\cellcolor{{orange!20}}{val}"
            return val

        print(f"{name} & {fmt(row['Front'])} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
              f"{highlight('Score_Prod')} & {highlight('Score_Mean')} & "
              f"{highlight('Score_Max')} & {highlight('Score_Maj')}\\\\")
        print("\\hline")

    print("\\end{tabular}")
    print(f"\\caption{{{title}}}")
    print("\\end{table*}\n\n")


# --- Average table ---
print_table(
    "Averaged MRR results over 8 views on the BFF dataset, showing performance across increasing levels of perspective error for all models.",
    avg_results
)

# --- STEP 6: Print one table per backbone ---
for backbone, per_data in all_backbones.items():
    print_table(
        f"MRR results for backbone \\texttt{{{backbone}}} across perspective error levels.",
        per_data
    )
