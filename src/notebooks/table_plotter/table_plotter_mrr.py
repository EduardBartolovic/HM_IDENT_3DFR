import re
import numpy as np
from collections import defaultdict


def generate_mrr_table(log_file: str, mode="gain", name="???", prec=3):
    """
    Parses a BFF evaluation log file and generates a LaTeX table.

    Args:
        name: dataset name
        log_file (str): Path to the log file.
        mode (str): "gain" to display gains vs Front, "absolute" to show absolute MRR.
    """
    # --- Load log lines ---
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
        "hyperface50k": r"Arcface-R50\textsubscript{HyperFace50K} ",
    }

    backbone = "Unknown"
    results = []
    skipnext = False

    # --- Parse log ---
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
                backbone_raw = (backbone_match.group(1).replace("_", "-")
                                .replace("test-rgb-bff-crop5-emb-", "")
                                .replace("test-nersemble-crop5-v15-emb-", "")
                                .replace("test-vox2train-crop5-v15-emb-", "")
                                .replace("test-vox2train-crop5d15-v15-emb-", "")
                                .replace("test-vox2train-crop5d10-v15-emb-", "")
                                .replace("test-vox2train-crop5d05-v15-emb-", "")
                                .replace("test-vox2train-crop5d03-v15-emb-", ""))

                backbone = backbone_map.get(backbone_raw, backbone_raw)
            continue

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

    # --- Prepare LaTeX table ---
    def fmt(v, prec=3):
        if prec == 3:
            return f"{v:.3f}" if not np.isnan(v) else "N/A"
        else:
            return f"{v:.2f}" if not np.isnan(v) else "N/A"

    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{2pt}")
    print("\\renewcommand{\\arraystretch}{0.98} % tighter rows")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Average} & \\multicolumn{1}{c|}{\\textbf{Pose}} & \\multicolumn{4}{c|}{\\textbf{Pose-Score}} \\\\")
    print("\\cline{5-8}")
    print("& \\textbf{Only} & \\textbf{Pooling} & \\textbf{Concat} & "
          "\\textbf{Prod} & \\textbf{Mean} & \\textbf{Max} & \\textbf{Maj} \\\\")
    print("\\hline")

    avg_fronts, avg_gains = [], defaultdict(list)

    for backbone, avg, gains in results:
        base = avg["Front"]
        avg_fronts.append(base)
        display_vals = gains if mode == "gain" else {k: avg[k] for k in avg if k != "Front"}

        for k, v in display_vals.items():
            if not np.isnan(v):
                avg_gains[k].append(v)

        valid_vals = {k: v for k, v in display_vals.items() if not np.isnan(v)}
        max_val = max(valid_vals.values()) if valid_vals else np.nan
        min_val = min(valid_vals.values()) if valid_vals else np.nan
        best_keys = [k for k, v in valid_vals.items() if v == max_val] if valid_vals else []
        worst_keys = [k for k, v in valid_vals.items() if v == min_val] if valid_vals else []

        def highlight(k):
            if k not in display_vals or np.isnan(display_vals[k]):
                return "N/A"
            val = fmt(display_vals[k], prec=prec)
            if k in best_keys:
                return f"\\cellcolor{{blue!15}}{{{val}}}"
            elif k in worst_keys:
                return f"\\cellcolor{{orange!20}}{{{val}}}"
            return val

        # Note the swapped columns: Average (Concat_Mean) <-> Pose (Concat)
        print(f"{backbone} & {fmt(avg['Front'], prec=prec)} & {highlight('Concat_Mean')} & {highlight('Concat')} & "
              f"{highlight('Score_Prod')} & {highlight('Score_Mean')} & "
              f"{highlight('Score_Max')} & {highlight('Score_Maj')}\\\\")
        print("\\hline")

    # --- Average row ---
    avg_vals = {k: np.nanmean(avg_gains[k]) for k in avg_gains}
    valid_avg_vals = {k: v for k, v in avg_vals.items() if not np.isnan(v)}
    max_val = max(valid_avg_vals.values())
    min_val = min(valid_avg_vals.values())
    best_keys = [k for k, v in valid_avg_vals.items() if v == max_val]
    worst_keys = [k for k, v in valid_avg_vals.items() if v == min_val]

    def highlight_avg(k):
        if k not in avg_vals or np.isnan(avg_vals[k]):
            return "N/A"
        val = fmt(avg_vals[k], prec=prec)
        if k in best_keys:
            return f"\\cellcolor{{blue!15}}{{{val}}}"
        elif k in worst_keys:
            return f"\\cellcolor{{orange!20}}{{{val}}}"
        return val

    print("\\hline")
    print("\\textbf{Average} & "
          f"{fmt(np.nanmean(avg_fronts), prec=prec)} & "
          f"{highlight_avg('Concat_Mean')} & {highlight_avg('Concat')} & "
          f"{highlight_avg('Score_Prod')} & {highlight_avg('Score_Mean')} & "
          f"{highlight_avg('Score_Max')} & {highlight_avg('Score_Maj')} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{" + ("Absolute MRR values" if mode == "absolute" else "Absolute MRR gains") +
          f" of multiview strategies compared to single-front view on the {name} dataset. Blue marks the best, orange the worst MV method." +"}")
    print("\\label{tab:" + f"{name}" + "}")
    print("\\end{table}")


#generate_mrr_table("test_BFF.txt", mode="gain", name="3D-BFF")
#generate_mrr_table("test_nersemble.txt", mode="gain", name="Nersemble")
#generate_mrr_table("test_vox2train.txt", mode="absolute", name="VoxCeleb2", prec=2)
#generate_mrr_table("test_vox2train_d15.txt", mode="absolute", name="VoxCeleb2d15", prec=2)
#generate_mrr_table("test_vox2train_d10.txt", mode="absolute", name="VoxCeleb2d10", prec=2)
#generate_mrr_table("test_vox2train_d05.txt", mode="absolute", name="VoxCeleb2d05", prec=2)
generate_mrr_table("test_vox2train_d03.txt", mode="absolute", name="VoxCeleb2d05", prec=2)

