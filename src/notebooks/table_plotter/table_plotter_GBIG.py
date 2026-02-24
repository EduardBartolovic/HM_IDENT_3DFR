import re
import numpy as np
from collections import defaultdict

def generate_gbig_table(log_file: str, mode="gain", name="???", prec=3):
    """
    Parses a BFF evaluation log file and generates a LaTeX table using GBIG.

    Args:
        name: dataset name
        log_file (str): Path to the log file.
        mode (str): "gain" to display gains vs Front, "absolute" to show absolute GBIG.
    """

    with open(log_file, "r") as f:
        lines = f.readlines()

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
        "hyperface50k": r"Arcface-R50\textsubscript{HyperFace50K}",
    }

    backbone = "Unknown"
    results = []
    skipnext = False

    for line in lines:
        if skipnext:
            skipnext = False
            continue

        line = line.strip()

        if "5F" in line or "5R" in line:
            skipnext = True
            continue

        if line.startswith("Perform 1:N Evaluation on"):
            backbone_match = re.search(r"/([^/]+)$", line.split(" with")[0])
            if backbone_match:
                backbone_raw = backbone_match.group(1)\
                    .replace("_", "-")\
                    .replace("test-rgb-bff-crop5-emb-", "")\
                    .replace("test-nersemble-crop5-v15-emb-", "")\
                    .replace("test-vox2train-crop5-v15-emb-", "")
                backbone = backbone_map.get(backbone_raw, backbone_raw)
            continue

        if "Front RR1:" in line:

            metrics = re.search(
                r"Front RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
                r"Concat RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
                r"Concat_Mean RR1:\s*[\d\.NA]+\s*MRR:\s*[\d\.NA]+\s*GBIG:\s*([\d\.NA]+).*?"
                r".*?Score_prod MRR:\s*[\d\.NA]+.*?"
                r"Score_mean MRR:\s*[\d\.NA]+.*?"
                r"Score_max MRR:\s*[\d\.NA]+.*?"
                r"Score_maj MRR:\s*[\d\.NA]+",
                line
            )

            if not metrics:
                continue

            vals = metrics.groups()

            avg = {
                "Front": float(vals[0]) if vals[0] != "N/A" else np.nan,
                "Concat": float(vals[1]) if vals[1] != "N/A" else np.nan,
                "Concat_Mean": float(vals[2]) if vals[2] != "N/A" else np.nan,
            }

            base = avg["Front"]
            gains = {k: avg[k] - base for k in avg if k != "Front"}

            results.append((backbone, avg, gains))

    def fmt(v, prec=3):
        return f"{v:.{prec}f}" if not np.isnan(v) else "N/A"

    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{2pt}")
    print("\\renewcommand{\\arraystretch}{0.98}")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Concat Mean} & \\textbf{Concat} \\\\")
    print("\\hline")

    avg_fronts, avg_gains = [], defaultdict(list)

    for backbone, avg, gains in results:

        base = avg["Front"]
        avg_fronts.append(base)

        display_vals = gains if mode == "gain" else {k: avg[k] for k in avg if k != "Front"}

        for k, v in display_vals.items():
            if not np.isnan(v):
                avg_gains[k].append(v)

        print(f"{backbone} & {fmt(avg['Front'], prec)} & "
              f"{fmt(display_vals.get('Concat_Mean', np.nan), prec)} & "
              f"{fmt(display_vals.get('Concat', np.nan), prec)}\\\\")
        print("\\hline")

    print("\\textbf{Average} & "
          f"{fmt(np.nanmean(avg_fronts), prec)} & "
          f"{fmt(np.nanmean(avg_gains['Concat_Mean']), prec)} & "
          f"{fmt(np.nanmean(avg_gains['Concat']), prec)} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{" +
          ("Absolute GBIG values" if mode == "absolute" else "Absolute GBIG gains") +
          f" of multiview strategies compared to single-front view on the {name} dataset." +
          "}")
    print("\\label{tab:" + f"{name}_GBIG" + "}")
    print("\\end{table}")



generate_gbig_table("test_BFF.txt", mode="gain", name="3D-BFF")
#generate_gbig_table("test_nersemble.txt", mode="gain", name="Nersemble")
#generate_gbig_table("test_vox2train.txt", mode="absolute", name="VoxCeleb2", prec=2)