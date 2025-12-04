import re
import numpy as np

# --- Load log ---
with open("E:\\Download\\PIE_GBIG.txt", "r") as f:
    log = f.read()


def fmt(v):
    return f"{v:.2f}" if not np.isnan(v) else "N/A"

# --- Backbone name mapping ---
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

# --- Split by backbone ---
backbone_blocks = re.split(r"=+\nLoading ONLY Backbone Checkpoint", log)
results = []

for block in backbone_blocks[1:]:
    backbone_match = re.search(r"[\\/](?:[^\\/]+)\.(pt|pth)", block)
    backbone = re.search(r"([^\\/]+)\.(?:pt|pth)", backbone_match.group(0)).group(1).replace("_", "-") if backbone_match else "Unknown"
    backbone_label = backbone_map.get(backbone, backbone)

    # Extract multipie dataset section
    dataset_match = re.search(r"Perform.*?multipie_crop8.*?(?=Perform|\Z)", block, flags=re.S)
    if not dataset_match:
        continue
    dataset_block = dataset_match.group(0)

    # --- Methods to extract ---
    methods = ["Front", "Concat", "Concat_Mean", "Score_prod", "Score_mean", "Score_max", "Score_maj"]
    all_vals = {}

    for m in methods:
        pattern = rf"(?<!\w){m}(?![_\w])[^|]*GBIG:\s*([0-9\.Ee+-]+)"
        matches = re.findall(pattern, dataset_block)
        if matches:
            all_vals[m] = float(matches[-1])

    if not all_vals:
        continue

    results.append((backbone_label, all_vals))

# --- Generate LaTeX table ---
print("\\begin{table*}[t]")
print("\\centering")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Concat} & \\textbf{Concat-Mean} & \\textbf{Score-Prod} & \\textbf{Score-Mean} & \\textbf{Score-Max} & \\textbf{Score-Maj} \\\\")
print("\\hline")

for backbone, vals in results:
    valid_vals = {k: v for k, v in vals.items() if not np.isnan(v)}
    best_key = max(valid_vals, key=valid_vals.get) if valid_vals else None

    def highlight(k):
        if k not in vals:
            return "N/A"
        v = vals[k]
        val = fmt(v)
        return f"\\cellcolor{{blue!15}}{{{val}}}" if k == best_key else val

    print(f"{backbone} & {highlight('Front')} & {highlight('Concat')} & {highlight('Concat_Mean')} & "
          f"{highlight('Score_prod')} & {highlight('Score_mean')} & {highlight('Score_max')} & {highlight('Score_maj')}\\\\")
    print("\\hline")

print("\\end{tabular}")
print("\\caption{Absolute GBIG scores for the Multipie dataset including score-based fusion methods.}")
print("\\label{tab:multipie}")
print("\\end{table*}")
