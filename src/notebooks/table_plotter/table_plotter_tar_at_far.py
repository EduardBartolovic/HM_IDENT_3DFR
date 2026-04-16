import re
import numpy as np
from collections import defaultdict


def generate_mrr_table(log_file: str, mode="gain", name="???", prec=3):
    """
    Parses a evaluation log file and generates a LaTeX table.

    Args:
        prec:
        name: dataset name
        log_file (str): Path to the log file.
        mode (str): "gain" to display gains vs Front, "absolute" to show absolute MRR.
    """
    # --- Load log lines ---
    with open(log_file, "r") as f:
        lines = f.readlines()

    backbone_map = {
        "ms1mv3-r18": r"Arcface-R18\textsubscript{MSM1V3}",
        "ms1mv3-r50": r"Arcface-R50\textsubscript{MSM1V3}",
        "ms1mv3-r100": r"Arcface-R100\textsubscript{MSM1V3}",
        "glint-r18": r"Cosface-R18\textsubscript{Glint}",
        "glint-r50": r"Cosface-R50\textsubscript{Glint}",
        "glint-r100": r"Cosface-R100\textsubscript{Glint}",
        "facenet-casia": r"Facenet512\textsubscript{Casia-Webface}",
        "facenet-vgg": r"Facenet512\textsubscript{VGGFace}",
        "adaface-ms1mv3": r"Adaface+Aroface\textsubscript{MSM1V3}",
        "adaface-webface12m": r"Adaface+Aroface\textsubscript{Web12M}",
        "edgeface-xs": r"EdgeFace-XS\textsubscript{Web12M}",
        "hyperface10k": r"Arcface-R50\textsubscript{HyperFace10K} ",
        "hyperface50k": r"Arcface-R50\textsubscript{HyperFace50K} ",
        "swinface": r"SwinFace\textsubscript{MSM1V3} ",
        "vit": r"Adaface-ViT\textsubscript{Web4M} ",
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
        if "ms1mv3_r50" in line:  #"5F" in line or "5R" in line: # "hyperface50k" in line or
            skipnext = True
            continue

        if line.startswith("Perform 1:N Evaluation on"):
            backbone_match = re.search(r"/([^/]+)$", line.split(" with")[0])
            if backbone_match:
                backbone_raw = (backbone_match.group(1).replace("_", "-")
                                .replace("test-rgb-bff-crop5-emb-", "")
                                .replace("test-rgb-bff-crop5E01-emb-", "")
                                .replace("test-rgb-bff-crop5E02-emb-", "")
                                .replace("test-rgb-bff-crop5E03-emb-", "")
                                .replace("test-rgb-bff-crop5E04-emb-", "")
                                .replace("test-rgb-ff-crop5-emb-", "")
                                .replace("test-rgb-ff-crop5E01-emb-", "")
                                .replace("test-rgb-ff-crop5E02-emb-", "")
                                .replace("test-rgb-ff-crop5E03-emb-", "")
                                .replace("test-rgb-ff-crop5E04-emb-", "")
                                .replace("test-rgb-ff-crop5E06-emb-", "")
                                .replace("test-rgb-ff-crop5E08-emb-", "")
                                .replace("test-rgb-ff-crop5E16-emb-", "")
                                .replace("test-rgb-ff-crop5E32-emb-", "")
                                .replace("test-nersemble-crop5-v15-emb-", "")
                                .replace("test-nersemble-crop5R-v15-emb-", "")
                                .replace("test-nersemble-crop5U-v15-emb-", "")
                                .replace("test-nersemble-crop5F-v15-emb-", "")
                                .replace("test-vox2train-crop5-v15-emb-", "")
                                .replace("test-vox2train-crop5F-v15-emb-", "")
                                .replace("test-vox2train-crop5d15-v15-emb-", "")
                                .replace("test-vox2train-crop5d10-v15-emb-", "")
                                .replace("test-vox2train-crop5d05-v15-emb-", "")
                                .replace("test-vox2train-crop5d03-v15-emb-", ""))

                backbone = backbone_map.get(backbone_raw, backbone_raw)
            continue

        if "TAR@FAR1e-6" in line:
            def extract(pattern):
                m = re.search(pattern, line)
                return float(m.group(1)) if m else np.nan

            tar = {
                "Front": extract(r"Front .*?TAR@FAR1e-5:\s*([\d\.]+)"),
                "Concat": extract(r"Concat .*?TAR@FAR1e-5:\s*([\d\.]+)"),
                "Concat_Mean": extract(r"Concat_Mean .*?TAR@FAR1e-5:\s*([\d\.]+)"),
                "Score_Prod": extract(r"Score_prod .*?TAR@FAR1e-5:\s*([\d\.]+)"),
                "Score_Max": extract(r"Score_max .*?TAR@FAR1e-5:\s*([\d\.]+)"),
                #"Score_Maj": extract(r"Score_maj .*?TAR@FAR1e-6:\s*([\d\.]+)"),
            }

            base = tar["Front"]
            gains = {k: tar[k] - base for k in tar if k != "Front"}

            results.append((backbone, tar, gains))

    # --- Prepare LaTeX table ---
    def fmt(v, prec=3):
        if np.isnan(v):
            return "N/A"
        if v == 100:
            return "100"
        if prec == 3:
            return f"{v:.3f}"
        else:
            return f"{v:.2f}"

    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{2pt}")
    print("\\renewcommand{\\arraystretch}{0.98} % tighter rows")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Backbone} & \\textbf{Front} & \\textbf{Average} & \\multicolumn{1}{c|}{\\textbf{Pose}} & \\multicolumn{2}{c|}{\\textbf{Pose-Score}} \\\\")
    print("\\cline{5-6}")
    print("& \\textbf{Only} & \\textbf{Pooling} & \\textbf{Concat} & "
          "\\textbf{Prod} & \\textbf{Max} \\\\")
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

        print(f"{backbone} & {fmt(avg['Front'], prec=prec)} & {highlight('Concat_Mean')} & {highlight('Concat')} & "
              f"{highlight('Score_Prod')} & "
              f"{highlight('Score_Max')}\\\\")
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
          f"{highlight_avg('Score_Prod')} & "
          f"{highlight_avg('Score_Max')} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{ \\textbf{" + name + ":} " +
          ("TAR@FAR=1e-5" if mode == "absolute" else "TAR@FAR=1e-5 gains") +
          " of multiview strategies compared to single-front view. Blue marks the best, orange the worst MV method." + "}")
    print("\\label{tab:" + f"{name}" + "}")
    print("\\end{table}")


if True:
    generate_mrr_table("ffE00.txt", mode="absolute", name="3D-FFE00", prec=2)
    generate_mrr_table("ffE01.txt", mode="absolute", name="3D-FFE01", prec=2)
    generate_mrr_table("ffE02.txt", mode="absolute", name="3D-FFE02", prec=2)
    generate_mrr_table("ffE03.txt", mode="absolute", name="3D-FFE03", prec=2)
    generate_mrr_table("ffE04.txt", mode="absolute", name="3D-FFE04", prec=2)
    generate_mrr_table("ffE06.txt", mode="absolute", name="3D-FFE06", prec=2)
    exit()
    generate_mrr_table("ffE08.txt", mode="absolute", name="3D-FFE08", prec=2)
    generate_mrr_table("ffE16.txt", mode="absolute", name="3D-FFE16", prec=2)
    generate_mrr_table("ffE32.txt", mode="absolute", name="3D-FFE32", prec=2)
if False:
    generate_mrr_table("nersemble.txt", mode="absolute", name="Nersemble")
    generate_mrr_table("nersembleR.txt", mode="absolute", name="NersembleR")
    generate_mrr_table("nersembleU.txt", mode="absolute", name="NersembleU")
if False:
    generate_mrr_table("vox2train-reduced.txt", mode="absolute", name="VoxCeleb2", prec=2)
    print("  ")
    generate_mrr_table("vox2trainR-reduced.txt", mode="absolute", name="VoxCeleb2", prec=2)
#generate_mrr_table("test_vox2train_d15.txt", mode="absolute", name="VoxCeleb2d15", prec=2)
#generate_mrr_table("test_vox2train_d10.txt", mode="absolute", name="VoxCeleb2d10", prec=2)
#generate_mrr_table("test_vox2train_d05.txt", mode="absolute", name="VoxCeleb2d05", prec=2)
#generate_mrr_table("test_vox2train_d03.txt", mode="absolute", name="VoxCeleb2d05", prec=2)
#generate_mrr_table("test_vox2trainF.txt", mode="absolute", name="VoxCeleb2", prec=2)

