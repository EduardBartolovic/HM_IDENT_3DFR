import re
import numpy as np
from collections import defaultdict

# Load log text
with open("test_nersemble.txt", "r") as f:
    lines = f.readlines()

mode = "gain"  # options: "absolute" or "gain"


def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "N/A"


backbone_map = {
    "ms1mv3-r18": r"Arcface-R18\textsubscript{MSMV3}",
    "ms1mv3-r50": r"Arcface-R50\textsubscript{MSMV3}",
    "ms1mv3-r100": r"Arcface-R100\textsubscript{MSMV3}",
    "glint-r18": r"Cosface-R18\textsubscript{Glint}",
    "glint-r50": r"Cosface-R50\textsubscript{Glint}",
    "glint-r100": r"Cosface-R100\textsubscript{Glint}",
    "facenet-casia": r"Facenet512\textsubscript{Casia-Webface}",
    "facenet-vgg": r"Facenet512\textsubscript{VGGFace}",
    "adaface-ms1mv3": r"Adaface+Aroface\textsubscript{MSMV3}",
    "adaface-webface12m": r"Adaface+Aroface\textsubscript{Web12M}",
    "edgeface-xs": r"EdgeFace-XS\textsubscript{Web12M}",
    "hyperface50k": r"Arcface-R50\textsubscript{HyperFace50K} ",
}

backbone = "Unknown"
results = []

skipnext = False
for line in lines:
    if skipnext:
        skipnext=False
        continue
    line = line.strip()

    if "5F" in line:
        skipnext=True
        continue

    if "5R" in line:
        skipnext=True
        continue

    # Detect lines with model paths
    if line.startswith("Perform 1:N Evaluation on"):
        backbone_match = re.search(r"/([^/]+)$", line.split(" with")[0])  # take path before "with"
        if backbone_match:
            backbone_raw = backbone_match.group(1).replace("_", "-").replace("test-nersemble-crop5-v15-emb-", "")
            backbone = backbone_map.get(backbone_raw, backbone_raw)
        continue  # next line contains metrics

    # If this line contains metrics, parse them using last seen backbone
    if "Front RR1:" in line:
        metrics = re.search(
            r"Front RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
            r"Concat RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
            r"Concat_Mean RR1:\s*([\d\.NA]+)\s*MRR:\s*([\d\.NA]+).*?"
            r"Score_prod MRR:\s*([\d\.NA]+).*?"
            r"Score_mean MRR:\s*([\d\.NA]+).*?"
            r"Score_max MRR:\s*([\d\.NA]+).*?"
            r"Score_maj MRR:\s*([\d\.NA]+)", line
        )

        if not metrics:
            continue

        vals = metrics.groups()
        avg = {
            "Front": float(vals[1]) if vals[1] != "N/A" else np.nan,
            "Concat": float(vals[3]) if vals[3] != "N/A" else np.nan,
            "Concat_Mean": float(vals[5]) if vals[5] != "N/A" else np.nan,
            "Score_Prod": float(vals[6]) if vals[6] != "N/A" else np.nan,
            "Score_Mean": float(vals[7]) if vals[7] != "N/A" else np.nan,
            "Score_Max": float(vals[8]) if vals[8] != "N/A" else np.nan,
            "Score_Maj": float(vals[9]) if vals[9] != "N/A" else np.nan,
        }

        base = avg["Front"]
        gains = {k: avg[k] - base for k in avg.keys() if k != "Front"}

        results.append((backbone, avg, gains))

# --- STEP 2: Generate LaTeX table (unchanged) ---
print("\\begin{table}[ht]")
print("\\centering")
print("\\scriptsize")
print("\\setlength{\\tabcolsep}{2pt}")
print("\\renewcommand{\\arraystretch}{0.98} % tighter rows")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\multicolumn{1}{c|}{\\textbf{Pose}} & \\multicolumn{1}{c|}{\\textbf{Average}} & \\multicolumn{4}{c|}{\\textbf{Score}} \\\\")
print("\\cline{5-8}")
print("& \\textbf{Only} & \\textbf{Match} & \\textbf{Pooling} & "
      "\\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\textbf{Maj} \\\\")
print("\\hline")

avg_fronts, avg_gains = [], defaultdict(list)

for backbone, avg, gains in results:
    base = avg["Front"]
    avg_fronts.append(base)

    # Select which values to display in table
    if mode == "gain":
        display_vals = gains  # show gains vs Front
    else:
        display_vals = {k: avg[k] for k in avg if k != "Front"}  # show absolute MRR

    # Collect for average row
    for k, v in display_vals.items():
        if not np.isnan(v):
            avg_gains[k].append(v)

    # Highlight best/worst (based on values in the row)
    valid_vals = {k: v for k, v in display_vals.items() if not np.isnan(v)}
    if valid_vals:
        max_val = max(valid_vals.values())
        min_val = min(valid_vals.values())
        best_keys = [k for k, v in valid_vals.items() if v == max_val]
        worst_keys = [k for k, v in valid_vals.items() if v == min_val]
    else:
        best_keys, worst_keys = [], []


    def highlight(k):
        if k not in display_vals or np.isnan(display_vals[k]):
            return "N/A"
        val = fmt(display_vals[k])
        if k in best_keys:
            return f"\\cellcolor{{blue!15}}{{{val}}}"
        elif k in worst_keys:
            return f"\\cellcolor{{orange!20}}{{{val}}}"
        return val

    print(f"{backbone} & {fmt(avg['Front'])} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Score_Prod')} & {highlight('Score_Mean')} & "
          f"{highlight('Score_Max')} & {highlight('Score_Maj')}\\\\")
    print("\\hline")

# --- Average row with highlighting ---
avg_vals = {
    "Concat": np.nanmean(avg_gains['Concat']),
    "Concat_Mean": np.nanmean(avg_gains['Concat_Mean']),
    "Score_Prod": np.nanmean(avg_gains['Score_Prod']),
    "Score_Mean": np.nanmean(avg_gains['Score_Mean']),
    "Score_Max": np.nanmean(avg_gains['Score_Max']),
    "Score_Maj": np.nanmean(avg_gains['Score_Maj']),
}

# Determine best/worst for average row
valid_avg_vals = {k: v for k, v in avg_vals.items() if not np.isnan(v)}
max_val = max(valid_avg_vals.values())
min_val = min(valid_avg_vals.values())
best_keys = [k for k, v in valid_avg_vals.items() if v == max_val]
worst_keys = [k for k, v in valid_avg_vals.items() if v == min_val]

def highlight_avg(k):
    if k not in avg_vals or np.isnan(avg_vals[k]):
        return "N/A"
    val = fmt(avg_vals[k])
    if k in best_keys:
        return f"\\cellcolor{{blue!15}}{{{val}}}"
    elif k in worst_keys:
        return f"\\cellcolor{{orange!20}}{{{val}}}"
    return val

print("\\hline")
print("\\textbf{Average} & "
      f"{fmt(np.nanmean(avg_fronts))} & "
      f"{highlight_avg('Concat')} & "
      f"{highlight_avg('Concat_Mean')} & "
      f"{highlight_avg('Score_Prod')} & "
      f"{highlight_avg('Score_Mean')} & "
      f"{highlight_avg('Score_Max')} & "
      f"{highlight_avg('Score_Maj')} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{" + ("Absolute MRR values" if mode == "absolute" else "Absolute MRR gains") + " of multiview strategies compared to single-front view on the Nersemble dataset. Blue marks the best, orange the worst MV method.}")
print("\\label{tab:???}")
print("\\end{table}")
