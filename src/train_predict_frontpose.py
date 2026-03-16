import numpy as np
import tempfile
import time
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.interpolate import LinearNDInterpolator

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.predictor.PoseFrontalizer import PoseFrontalizer, PoseFrontalizerWithPose, PoseFrontalizerWithPoseResidual
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset, split_with_shared_labels
from src.util.misc import colorstr
from util.utils import schedule_lr, AverageMeter
from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()


def mse_loss_normalized(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


def cosine_mse_loss(pred, target, mse_weight=0.3):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    mse = F.mse_loss(pred, target)
    return cos + mse_weight * mse


def cosine_smoothl1_loss(pred, target, l1_weight=0.3):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    l1 = F.smooth_l1_loss(pred, target)

    return cos + l1_weight * l1


def validate(model, data_loader, device, with_pose):
    model.eval()

    cos_model_meter = AverageMeter()
    cos_baseline_meter = AverageMeter()

    with torch.no_grad():
        for embeddings, labels, scan_ids, true_poses, _ in data_loader:
            B, V, D = embeddings.shape

            rand_idx = torch.randint(0, V, (B,), device=embeddings.device)
            input_emb = embeddings[torch.arange(B), rand_idx].to(device)
            input_pose = true_poses[torch.arange(B), rand_idx].to(device) # (B, 2)

            # check where pose == (0, 0)
            is_front = (true_poses == 0).all(dim=-1)  # (B, V) boolean
            assert (is_front.sum(dim=1) == 1).all(), "Missing or multiple front views!"
            front_pose_index = is_front.long().argmax(dim=1)  # (B,)
            front_emb = embeddings[torch.arange(B), front_pose_index].to(device)

            if with_pose:
                pred_front = model(input_emb, input_pose)
            else:
                pred_front = model(input_emb)

            cos_model = F.cosine_similarity(
                F.normalize(pred_front, dim=-1),
                F.normalize(front_emb, dim=-1),
                dim=-1
            ).mean()

            # --- baseline: random view directly vs front ---
            cos_baseline = F.cosine_similarity(
                F.normalize(input_emb, dim=-1),
                F.normalize(front_emb, dim=-1),
                dim=-1
            ).mean()

            cos_model_meter.update(cos_model.item(), B)
            cos_baseline_meter.update(cos_baseline.item(), B)

    return {
        "cosine_model": cos_model_meter.avg,
        "cosine_baseline": cos_baseline_meter.avg,
        "cosine_gain": cos_model_meter.avg - cos_baseline_meter.avg
    }


def validate_with_pose_heatmap(model, data_loader, device, with_pose, prefix="val"):
    model.eval()

    # ------------------------------------------------------------
    # Collect all poses first (from first batch)
    # ------------------------------------------------------------
    sample_batch = next(iter(data_loader))
    _, _, _, true_poses_ref, _ = sample_batch
    poses_ref = true_poses_ref[0].cpu().numpy()  # (V, 2)

    pitches = np.unique(poses_ref[:, 0])
    yaws    = np.unique(poses_ref[:, 1])

    H, W = len(pitches), len(yaws)

    pitch_to_idx = {p: i for i, p in enumerate(pitches)}
    yaw_to_idx   = {y: j for j, y in enumerate(yaws)}

    # ------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------
    heat_model_sum = np.zeros((H, W), dtype=np.float32)
    heat_base_sum  = np.zeros((H, W), dtype=np.float32)
    heat_count     = np.zeros((H, W), dtype=np.int32)

    # ------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------
    with torch.no_grad():
        for embeddings, labels, scan_ids, true_poses, _ in data_loader:

            B, V, D = embeddings.shape
            embeddings = embeddings.to(device)
            true_poses = true_poses.to(device)

            # locate front view (0,0)
            is_front = (true_poses == 0).all(dim=-1)
            front_pose_index = is_front.long().argmax(dim=1)
            front_emb = embeddings[torch.arange(B), front_pose_index]

            # randomly pick one input view
            rand_idx = torch.randint(0, V, (B,), device=device)
            input_emb = embeddings[torch.arange(B), rand_idx]
            input_pose = true_poses[torch.arange(B), rand_idx]

            if with_pose:
                pred_front = model(input_emb, input_pose)
            else:
                pred_front = model(input_emb)

            cos_model = F.cosine_similarity(
                F.normalize(pred_front, dim=-1),
                F.normalize(front_emb, dim=-1),
                dim=-1
            )

            cos_baseline = F.cosine_similarity(
                F.normalize(input_emb, dim=-1),
                F.normalize(front_emb, dim=-1),
                dim=-1
            )

            # ----------------------------------------------------
            # Accumulate per pose
            # ----------------------------------------------------
            for b in range(B):
                pitch, yaw = input_pose[b].cpu().numpy()
                i = pitch_to_idx[int(pitch)]
                j = yaw_to_idx[int(yaw)]

                heat_model_sum[i, j] += cos_model[b].item()
                heat_base_sum[i, j]  += cos_baseline[b].item()
                heat_count[i, j]     += 1

    # ------------------------------------------------------------
    # Average
    # ------------------------------------------------------------
    heat_model = heat_model_sum / np.maximum(heat_count, 1)
    heat_base  = heat_base_sum  / np.maximum(heat_count, 1)
    heat_gain  = heat_model - heat_base

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    def plot_heatmap(data, title, filename):

        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 26,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        })

        # -------------------------------------------------------
        # scale values
        # -------------------------------------------------------
        data = data * 100

        # -------------------------------------------------------
        # build pose grid
        # -------------------------------------------------------
        grid_pitch, grid_yaw = np.meshgrid(pitches, yaws, indexing="ij")
        grid_points = np.stack([grid_pitch.ravel(), grid_yaw.ravel()], axis=1)

        # -------------------------------------------------------
        # extract valid samples
        # -------------------------------------------------------
        valid_mask = heat_count > 0

        sample_points = []
        sample_values = []

        H, W = data.shape
        for i in range(H):
            for j in range(W):
                if valid_mask[i, j]:
                    sample_points.append((pitches[i], yaws[j]))
                    sample_values.append(data[i, j])

        sample_points = np.array(sample_points)
        sample_values = np.array(sample_values)

        # -------------------------------------------------------
        # interpolate
        # -------------------------------------------------------
        interp = LinearNDInterpolator(sample_points, sample_values, fill_value=np.nan)

        heat_interp = interp(grid_points).reshape(H, W)

        # mask invalid areas (outside convex hull)
        heat_plot = np.ma.masked_where(np.isnan(heat_interp), heat_interp)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")

        vmax = np.nanmax(np.abs(heat_plot))
        vmin = np.nanmin(heat_plot)

        fig, ax = plt.subplots(figsize=(22, 20))

        sns.heatmap(
            heat_plot,
            ax=ax,
            xticklabels=yaws,
            yticklabels=pitches,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={
                "shrink": 0.8,
                "label": "Cosine Similarity ×100"
            }
        )

        for i in range(H):
            for j in range(W):
                if not valid_mask[i, j]:
                    continue

                val = data[i, j]
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )

        ax.set_title(title)
        ax.set_xlabel("Yaw")
        ax.set_ylabel("Pitch")

        for i, label in enumerate(ax.get_xticklabels()):
            if i % 5 != 0:
                label.set_visible(False)

        for i, label in enumerate(ax.get_yticklabels()):
            if i % 5 != 0:
                label.set_visible(False)

        #fig.savefig(f"{prefix}_{filename}.svg", format="svg", dpi=200, bbox_inches="tight")
        fig.savefig(f"{prefix}_{filename}.jpg", format="jpg", dpi=200, bbox_inches="tight")

    plot_heatmap(heat_model, "Model Cosine vs Front", "model")
    plot_heatmap(heat_base, "Baseline Cosine vs Front", f"baseline")
    plot_heatmap(heat_gain, "Model Gain over Baseline", f"gain")

    return heat_model, heat_base, heat_gain


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    LOG_ROOT = cfg['LOG_ROOT']

    PREDICTOR_NAME = cfg['PREDICTOR_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']
    LOSS_NAME = cfg.get("LOSS_NAME", "cosine")

    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # embedding dimension
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    # ===== ML FLOW Set up ============
    mlflow.set_tracking_uri(f'file:{LOG_ROOT}/mlruns')
    mlflow.set_experiment(RUN_NAME)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(RUN_NAME)
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_count = len(runs)
    else:
        run_count = 0

    with mlflow.start_run(run_name=f"{RUN_NAME}_[{run_count + 1}]") as run:
        mlflow.log_param('config', cfg)
        print(f"{RUN_NAME}_{run_count + 1} ; run_id:", run.info.run_id)
        full_dataset = EmbeddingDataset(os.path.join(DATA_ROOT, TRAIN_SET), disable_tqdm=False)
        train_dataset, val_dataset = split_with_shared_labels(
            full_dataset,
            val_ratio=0.1,
            seed=SEED
        )
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        val_dataset2 = EmbeddingDataset(os.path.join(DATA_ROOT, "rgb_monoffhq_crop25_emb-glint_r18"), disable_tqdm=False)
        print(f"Val2 samples: {len(val_dataset2)}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            prefetch_factor=4,
            persistent_workers=True,
            num_workers=NUM_WORKERS,
            drop_last=DROP_LAST,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=False,
            shuffle=False
        )
        val_loader2 = torch.utils.data.DataLoader(
            val_dataset2,
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=False,
            shuffle=False
        )

        # ======= Predictor =======
        predictor_dict = {'PoseFrontalizer': lambda: PoseFrontalizer(),
                          'PoseFrontalizerWithPose': lambda: PoseFrontalizerWithPose(),
                          'PoseFrontalizerWithPoseResidual': lambda: PoseFrontalizerWithPoseResidual()}

        predictor = predictor_dict[PREDICTOR_NAME]()
        predictor.to(DEVICE)
        if PREDICTOR_NAME == "PoseFrontalizerWithPose" or PREDICTOR_NAME == "PoseFrontalizerWithPoseResidual":
            dummy_emb = torch.zeros(BATCH_SIZE, EMBEDDING_SIZE).to(DEVICE)
            dummy_pose = torch.zeros(BATCH_SIZE, 2).to(DEVICE)
            model_stats_predictor = summary(
                predictor,
                input_data=(dummy_emb, dummy_pose),
                verbose=0
            )
        else:
            model_stats_predictor = summary(predictor, (BATCH_SIZE, EMBEDDING_SIZE), verbose=0)
        print(colorstr('magenta', str(model_stats_predictor)))
        print(colorstr('blue', f"{PREDICTOR_NAME} Backbone Generated"))
        print("=" * 60)

        # ======= Write Summaries =======
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with open(os.path.join(tmp_dir, 'Predictor_Summary.txt'), "w", encoding="utf-8") as f:
                f.write(str(model_stats_predictor))
            mlflow.log_artifacts(str(tmp_dir), artifact_path="ModelSummary")

        # ======= Optimizer Settings =======
        params_list = [{'params': predictor.parameters(), 'weight_decay': WEIGHT_DECAY}]
        OPTIMIZER_DICT = {'SGD': lambda: optim.SGD(params_list, lr=LR, momentum=MOMENTUM),
                          'ADAM': lambda: optim.Adam(params_list, lr=LR),
                          'ADAMW': lambda: optim.AdamW(params_list, lr=LR)}
        OPTIMIZER = OPTIMIZER_DICT[OPTIMIZER_NAME]()
        print(colorstr('magenta', OPTIMIZER))
        print("=" * 60)

        LOSS_DICT = {
            "cosine": cosine_loss,
            "mse_norm": mse_loss_normalized,
            "cosine_mse": cosine_mse_loss,
            "cosine_smoothl1": cosine_smoothl1_loss,
        }
        loss_fn = LOSS_DICT[LOSS_NAME]
        print(colorstr("magenta", f"Using loss: {LOSS_NAME}"))
        print("=" * 60)

        # load_checkpoint(BACKBONE.backbone_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        # print("=" * 60)

        # ======= Validation =======
        print("#" * 60)
        val_metrics = validate(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")
        validate_with_pose_heatmap(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose", prefix=str(0)+"bff")

        val_metrics2 = validate(predictor, val_loader2, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")
        validate_with_pose_heatmap(predictor, val_loader2, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose", prefix=str(0)+"mono")

        mlflow.log_metric('val_cosine_prediction', val_metrics["cosine_model"], step=0)
        mlflow.log_metric('val_cosine_baseline', val_metrics["cosine_baseline"], step=0)
        mlflow.log_metric('val_cosine_gain', val_metrics["cosine_gain"], step=0)

        print(colorstr(
            'bright_green',
            f'Val Model Cosine Similarity: {val_metrics["cosine_model"]:.4f} | '
            f'Val Baseline Cosine Similarity: {val_metrics["cosine_baseline"]:.4f} | '
            f'Val Gain: {val_metrics["cosine_gain"]:+.4f}'
            f'Val2 Model Cosine Similarity: {val_metrics2["cosine_model"]:.4f} | '
            f'Val2 Baseline Cosine Similarity: {val_metrics2["cosine_baseline"]:.4f} | '
            f'Val2 Gain: {val_metrics2["cosine_gain"]:+.4f}'
        ))

        print("#" * 60)

        # ======= train & early stopping parameters=======
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc
        batch = 0  # batch index
        for epoch in range(0, NUM_EPOCH):

            if epoch in STAGES:
                schedule_lr(OPTIMIZER, factor=2)

            predictor.train()

            # ======= Train Loop ========
            losses = AverageMeter()

            for step, (embeddings, labels, scan_ids, true_poses, _) in enumerate(tqdm(train_loader)):

                B, V, D = embeddings.shape

                rand_idx = torch.randint(0, V, (B,), device=embeddings.device)
                input_emb = embeddings[torch.arange(B), rand_idx].to(DEVICE) # (B, D)
                input_pose = true_poses[torch.arange(B), rand_idx].to(DEVICE) # (B, 2)

                # check where pose == (0, 0)
                is_front = (true_poses == 0).all(dim=-1)  # (B, V) boolean
                assert (is_front.sum(dim=1) == 1).all(), "Missing or multiple front views!"
                front_pose_index = is_front.long().argmax(dim=1)  # (B,)
                front_emb = embeddings[torch.arange(B), front_pose_index].to(DEVICE)

                if PREDICTOR_NAME == "PoseFrontalizerWithPose":
                    pred_front = predictor(input_emb, input_pose)
                else:
                    pred_front = predictor(input_emb)

                loss = loss_fn(pred_front, front_emb)
                losses.update(loss.item(), B)
                mlflow.log_metric('train_loss', losses.avg, step=epoch + 1)

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    print(colorstr('cyan',
                                   f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                   f'Training Loss {losses.avg:.4f}\t'))
                    print("=" * 60)
                batch += 1

            # ===== Validation =====
            val_metrics = validate(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")
            validate_with_pose_heatmap(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose", prefix=str(epoch))

            val_metrics2 = validate(predictor, val_loader2, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")
            validate_with_pose_heatmap(predictor, val_loader2, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose", prefix=str(0) + "mono")

            mlflow.log_metric('val_cosine_prediction', val_metrics["cosine_model"], step=epoch + 1)
            mlflow.log_metric('val_cosine_baseline', val_metrics["cosine_baseline"], step=epoch + 1)
            mlflow.log_metric('val_cosine_gain', val_metrics["cosine_gain"], step=epoch + 1)

            print(colorstr(
                'bright_green',
                f'Epoch {epoch + 1}/{NUM_EPOCH} | '
                f'Train Cosine Similarity Loss: {losses.avg:.4f} | '
                f'Val Model Cosine Similarity: {val_metrics["cosine_model"]:.4f} | '
                f'Val Baseline Cosine Similarity: {val_metrics["cosine_baseline"]:.4f} | '
                f'Val Gain: {val_metrics["cosine_gain"]:+.4f}'
                f'Val2 Model Cosine Similarity: {val_metrics2["cosine_model"]:.4f} | '
                f'Val2 Baseline Cosine Similarity: {val_metrics2["cosine_baseline"]:.4f} | '
                f'Val2 Gain: {val_metrics2["cosine_gain"]:+.4f}'
            ))
            print("#" * 60)

            time.sleep(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    main(cfg_yaml)
