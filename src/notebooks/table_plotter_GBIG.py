import re
import numpy as np
from collections import defaultdict

# Load log text
with open("E:\\Download\\BFF_GBIG.txt", "r") as f:
    log = f.read()

mode = "absolute"  # options: "absolute" or "gain"


def fmt(v):
    return f"{v:.2f}" if not np.isnan(v) else "N/A"


backbone_map = {
    "ms1mv3-arcface-r18-fp16": r"Arcface-R18\textsubscript{MS1MV3}",
    "ms1mv3-arcface-r50-fp16": r"Arcface-R50\textsubscript{MS1MV3}",
    "ms1mv3-arcface-r100-fp16": r"Arcface-R100\textsubscript{MS1MV3}",
    "glint-cosface-r18-fp16": r"Cosface-R18\textsubscript{Glint}",
    "glint-cosface-r50-fp16": r"Cosface-R50\textsubscript{Glint}",
    "glint-cosface-r100-fp16": r"Cosface-R100\textsubscript{Glint}",
    "facenet-casia-webface": r"Facenet512\textsubscript{Casia-Webface}",
    "facenet-vggface2": r"Facenet512\textsubscript{VGGFace}",
    "AdaFace-ARoFace-R100-MS1MV3": r"Adaface+Aroface\textsubscript{MS1MV3}",
    "AdaFace-ARoFace-R100-WebFace12M": r"Adaface+Aroface\textsubscript{WebFace12M}",
    "edgeface-xs-gamma-06": r"EdgeFace-XS\textsubscript{WebFace12M}",
}

# --- STEP 1: Parse log into backbone blocks ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)
results = []

for block in backbone_blocks[1:]:
    # Extract backbone name
    backbone_match = re.search(r"[\\/](?:[^\\/]+)\.(pt|pth)", block)
    if backbone_match:
        backbone_raw = re.search(r"([^\\/]+)\.(?:pt|pth)", backbone_match.group(0)).group(1).replace("_", "-")
        backbone = backbone_map.get(backbone_raw, backbone_raw)
    else:
        backbone = "Unknown"

    # --- Extract GBIG values per test set ---
    evals = re.findall(
        r"(\b(?:nersemble8|rgb_bff_crop8|vox2train_crop8|ytf_crop8|multipie_crop5)\b).*?"
        r"Front RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Concat RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Concat_Mean RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Score_prod MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Score_mean MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Score_max MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"Score_maj MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
        r"MV RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+)",
        block, flags=re.S,
    )

    if not evals:
        continue

    all_vals = defaultdict(list)
    for ev in evals:
        dataset = ev[0]
        all_vals["Front"].append(float(ev[1]) if ev[1] != "N/A" else np.nan)
        all_vals["Concat"].append(float(ev[2]) if ev[2] != "N/A" else np.nan)
        all_vals["Concat_Mean"].append(float(ev[3]) if ev[3] != "N/A" else np.nan)
        all_vals["Score_Prod"].append(float(ev[4]) if ev[4] != "N/A" else np.nan)
        all_vals["Score_Mean"].append(float(ev[5]) if ev[5] != "N/A" else np.nan)
        all_vals["Score_Max"].append(float(ev[6]) if ev[6] != "N/A" else np.nan)
        all_vals["Score_Maj"].append(float(ev[7]) if ev[7] != "N/A" else np.nan)
        all_vals["MV"].append(float(ev[8]) if ev[8] != "N/A" else np.nan)

    avg = {k: np.nanmean(v) for k, v in all_vals.items()}
    base = avg["Front"]
    gains = {k: avg[k] - base for k in avg.keys() if k != "Front"}
    results.append((backbone, avg, gains))

# --- STEP 2: Generate LaTeX table with Prod & Maj removed ---
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\setlength{\\tabcolsep}{3pt}")
print("\\renewcommand{\\arraystretch}{0.95}")  # tighter rows
print("\\begin{tabular}{|l|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\multicolumn{1}{c|}{\\textbf{Concat}} & "
      "\\multicolumn{1}{c|}{\\textbf{Average}} & \\multicolumn{2}{c|}{\\textbf{Score}} \\\\")
print("\\cline{5-6}")
print("& \\textbf{Only} & \\textbf{} & \\textbf{Pooling} & \\textbf{Mean} & \\textbf{Max} \\\\")
print("\\hline")

avg_fronts, avg_gains = [], defaultdict(list)

for backbone, avg, gains in results:
    base = avg["Front"]
    avg_fronts.append(base)

    if mode == "gain":
        for k in gains:
            avg_gains[k].append(gains[k])
    else:
        for k in avg:
            if k not in ["Front", "Score_Prod", "Score_Maj"]:
                avg_gains[k].append(avg[k])

    # Absolute mode highlighting
    valid_vals = {k: avg[k] for k in avg if k not in ["Front", "Score_Prod", "Score_Maj"] and not np.isnan(avg[k])}
    best_key = max(valid_vals, key=valid_vals.get) if valid_vals else None
    worst_key = min(valid_vals, key=valid_vals.get) if valid_vals else None

    def highlight(k):
        if k not in avg or np.isnan(avg[k]):
            return "N/A"
        val = fmt(avg[k])
        if k == best_key:
            return f"\\cellcolor{{blue!15}}{{{val}}}"
        elif k == worst_key:
            return f"\\cellcolor{{orange!20}}{{{val}}}"
        return val

    print(f"{backbone} & {fmt(avg['Front'])} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Score_Mean')} & {highlight('Score_Max')}\\\\")
    print("\\hline")

# --- Average row ---
print("\\hline")
print("\\textbf{Average} & "
      f"{np.nanmean(avg_fronts):.3f} & "
      f"{fmt(np.nanmean(avg_gains['Concat']))} & "
      f"{fmt(np.nanmean(avg_gains['Concat_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Max']))} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{" + ("Absolute GBIG values" if mode == "absolute" else "GBIG gains vs Front") + " on the dataset}")
print("\\label{tab:gbig_results}")
print("\\end{table}")
