import re
import numpy as np
from collections import defaultdict

# --- Load log file ---
with open("E:\\Download\\performace.txt", "r") as f:
    log = f.read()

mode = "absolute"  # or "gain"
def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"

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

# --- Split log into backbone sections ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)
results = []

for block in backbone_blocks[1:]:
    # Extract backbone filename
    backbone_match = re.search(r"[\\/](?:[^\\/]+)\.(pt|pth)", block)
    if not backbone_match:
        continue
    backbone_raw = re.search(r"([^\\/]+)\.(?:pt|pth)", backbone_match.group(0)).group(1).replace("_", "-")
    backbone = backbone_map.get(backbone_raw, backbone_raw)

    # Extract YTF AUC results
    ytf_match = re.search(
        r"ytf_crop8.*?Front-AUC/Acc:\s*([\d\.]+).*?"
        r"Concat-AUC/Acc:\s*([\d\.]+).*?"
        r"Concat_Mean-AUC/Acc:\s*([\d\.]+).*?"
        r"Concat_PCA-AUC:\s*([\d\.]+).*?"
        r"Score_sum-AUC:\s*([\d\.]+).*?"
        r"Score_prod-AUC:\s*([\d\.]+).*?"
        r"Score_mean-AUC:\s*([\d\.]+).*?"
        r"Score_max-AUC:\s*([\d\.]+).*?"
        r"MV-AUC/Acc:\s*([\d\.]+)",
        block, flags=re.S,
    )

    if not ytf_match:
        continue

    vals = [float(v) for v in ytf_match.groups()]
    keys = ["Front", "Concat", "Concat_Mean", "Concat_PCA",
            "Score_Sum", "Score_Prod", "Score_Mean", "Score_Max", "MV"]
    avg = dict(zip(keys, vals))
    base = avg["Front"]
    gains = {k: avg[k] - base for k in avg if k != "Front"}
    results.append((backbone, avg, gains))

# --- Output LaTeX table ---
print("\\begin{table*}[t]")
print("\\centering")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Concat} & \\textbf{Mean} & "
      "\\textbf{PCA} & \\multicolumn{4}{c|}{\\textbf{Score}} & \\textbf{MV} \\\\")
print("\\cline{5-9}")
print("& & & & & \\textbf{Sum} & \\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\\\")
print("\\hline")

avg_fronts, avg_gains = [], defaultdict(list)

for backbone, avg, gains in results:
    avg_fronts.append(avg["Front"])
    if mode == "gain":
        for k, v in gains.items(): avg_gains[k].append(v)
    else:
        for k, v in avg.items():
            if k != "Front": avg_gains[k].append(v)

    valid_vals = {k: avg[k] for k in avg if k != "Front" and not np.isnan(avg[k])}
    best_key = max(valid_vals, key=valid_vals.get)
    worst_key = min(valid_vals, key=valid_vals.get)

    def highlight(k):
        if np.isnan(avg[k]): return "N/A"
        val = fmt(avg[k])
        if k == best_key: return f"\\cellcolor{{blue!15}}{{{val}}}"
        elif k == worst_key: return f"\\cellcolor{{orange!20}}{{{val}}}"
        return val

    print(f"{backbone} & {fmt(avg['Front'])} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Concat_PCA')} & {highlight('Score_Sum')} & {highlight('Score_Prod')} & "
          f"{highlight('Score_Mean')} & {highlight('Score_Max')} & {highlight('MV')}\\\\")
    print("\\hline")

# --- Averages ---
print("\\textbf{Average} & "
      f"{fmt(np.nanmean(avg_fronts))} & "
      f"{fmt(np.nanmean(avg_gains['Concat']))} & "
      f"{fmt(np.nanmean(avg_gains['Concat_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Concat_PCA']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Sum']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Prod']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Mean']))} & "
      f"{fmt(np.nanmean(avg_gains['Score_Max']))} & "
      f"{fmt(np.nanmean(avg_gains['MV']))} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{" + ("Absolute AUC values" if mode == "absolute" else "AUC gains vs Front") + " on YTF.}")
print("\\label{tab:ytf_auc}")
print("\\end{table*}")
