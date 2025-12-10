import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.predictor.EmbeddingHPE import EmbeddingHPE
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset, split_with_shared_labels
from src.util.datapipeline.datasets import AFLW2000EMB
from src.util.misc import colorstr
from util.utils import warm_up_lr, schedule_lr, AverageMeter
from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def evaluate(model, data_loader, device, perspective_range):
    model.eval()
    total = 0
    yaw_error = 0.0
    pitch_error = 0.0

    for embs, r_label, cont_labels, name in data_loader:
        embs = embs.to(device)
        total += cont_labels.size(0)

        # ---- Ground truth in degrees ----
        p_gt_deg = cont_labels[:, 0].float() * 180 / np.pi  # pitch
        y_gt_deg = cont_labels[:, 1].float() * 180 / np.pi  # yaw

        # ---- Model prediction (B, 2): [yaw, pitch] ----
        pred = model(embs).cpu()
        yaw_pred = pred[:, 0]*perspective_range[1]
        pitch_pred = pred[:, 1]*perspective_range[1]

        # ---- Angular error (cyclic) ----
        pitch_error += torch.sum(torch.min(torch.stack([
            torch.abs(p_gt_deg - pitch_pred),
            torch.abs(pitch_pred + 360 - p_gt_deg),
            torch.abs(pitch_pred - 360 - p_gt_deg),
            torch.abs(pitch_pred + 180 - p_gt_deg),
            torch.abs(pitch_pred - 180 - p_gt_deg)
        ]), dim=0)[0])

        yaw_error += torch.sum(torch.min(torch.stack([
            torch.abs(y_gt_deg - yaw_pred),
            torch.abs(yaw_pred + 360 - y_gt_deg),
            torch.abs(yaw_pred - 360 - y_gt_deg),
            torch.abs(yaw_pred + 180 - y_gt_deg),
            torch.abs(yaw_pred - 180 - y_gt_deg)
        ]), dim=0)[0])

    print(colorstr('bright_green',
        f'Yaw: {yaw_error / total:.4f} '
        f'Pitch: {pitch_error / total:.4f} '
        f'MAE: {(yaw_error + pitch_error) / (total * 2):.4f}'
    ))



def validate(model, data_loader, device, perspective_range):
    model.eval()

    mse_meter = AverageMeter()
    mae_meter = AverageMeter()
    mae_pitch_meter = AverageMeter()
    mae_yaw_meter = AverageMeter()

    with torch.no_grad():
        for embeddings, labels, scan_ids, true_poses, path in iter(data_loader):
            B, V, D = embeddings.shape

            rand_idx = torch.randint(0, V, (B,), device=embeddings.device)
            embedding = embeddings[torch.arange(B), rand_idx].to(device)
            true_pose = true_poses[torch.arange(B), rand_idx].to(device).float()

            pred_pose = model(embedding)

            # --- Compute errors ---
            mse = F.mse_loss(pred_pose, true_pose)
            mae = F.l1_loss(pred_pose, true_pose)
            pitch_mae = torch.mean(torch.abs(pred_pose[:, 0] - true_pose[:, 0]))
            yaw_mae   = torch.mean(torch.abs(pred_pose[:, 1] - true_pose[:, 1]))

            # --- Update meters ---
            batch_size = embedding.size(0)
            mse_meter.update(mse.item(), batch_size)
            mae_meter.update(mae.item(), batch_size)
            mae_pitch_meter.update(pitch_mae.item(), batch_size)
            mae_yaw_meter.update(yaw_mae.item(), batch_size)

    return {
        "mse": mse_meter.avg,
        "mae": mae_meter.avg,
        "mae_pitch": mae_pitch_meter.avg,
        "mae_yaw": mae_yaw_meter.avg
    }

def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']

    PREDICTOR_NAME = cfg['PREDICTOR_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']

    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # embedding dimension
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    PATIENCE = cfg.get('PATIENCE', 10)
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
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
        full_dataset = EmbeddingDataset(os.path.join(DATA_ROOT, TRAIN_SET))
        dataset_size = len(full_dataset)
        split = int(0.9 * dataset_size)
        #train_dataset, val_dataset = torch.utils.data.random_split(
        #    full_dataset,
        #    [split, dataset_size - split],
        #    generator=torch.Generator().manual_seed(SEED)
        #)

        train_dataset, val_dataset = split_with_shared_labels(
            full_dataset,
            val_ratio=0.1,
            seed=SEED
        )

        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            drop_last=DROP_LAST,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            drop_last=False,
            shuffle=False
        )

        # ======= Predictor =======
        predictor_dict = {'EmbeddingHPE': lambda: EmbeddingHPE()}

        predictor = predictor_dict[PREDICTOR_NAME]()
        predictor.to(DEVICE)
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

        # load_checkpoint(BACKBONE.backbone_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        # print("=" * 60)

        # ======= Validation =======
        # TODO INPUT Embeddings+ POSe Normalisieren
        # TODO YAW und PITCH GETRENNT mindetsens eval
        # TODO: Mehrere Modelle testen.
        # TODO echten Posendatensatz

        print("#" * 60)
        predictor.eval()
        val_metrics = validate(predictor, val_loader, DEVICE, perspective_range)

        mlflow.log_metric('val_mse', val_metrics["mse"], step=0)
        mlflow.log_metric('val_mae', val_metrics["mae"], step=0)
        mlflow.log_metric('val_mae_pitch', val_metrics["mae_pitch"], step=0)
        mlflow.log_metric('val_mae_yaw', val_metrics["mae_yaw"], step=0)

        print(colorstr(
            'bright_green',
            f'Epoch 0 /{NUM_EPOCH} | '
            f'MSE: {val_metrics["mse"]:.4f} | '
            f'MAE: {val_metrics["mae"]:.4f} | '
            f'Pitch MAE: {val_metrics["mae_pitch"]:.4f} | '
            f'Yaw MAE: {val_metrics["mae_yaw"]:.4f}'
        ))

        #aflw_dataset = AFLW2000EMB("C:\\Users\\Eduard\\Desktop\\Face\\dataset11\\AFLW2000-3D\\")
        #aflw_loader = torch.utils.data.DataLoader(
        #    dataset=aflw_dataset,
        #    batch_size=32,
        #    num_workers=1,
        #    pin_memory=True
        #)
        #evaluate(model=predictor, data_loader=aflw_loader, device=DEVICE, perspective_range=perspective_range)
        print("#" * 60)


        # ======= train & early stopping parameters=======
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc
        NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
        NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
        batch = 0  # batch index
        for epoch in range(0, NUM_EPOCH):

            if epoch in STAGES:
                schedule_lr(OPTIMIZER, factor=2)

            predictor.train()

            # =========== Train Loop ========
            losses = AverageMeter()
            for step, (embeddings, labels, scan_ids, true_poses, path) in enumerate(tqdm(iter(train_loader))):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                B, V, D = embeddings.shape
                rand_idx = torch.randint(0, V, (B,), device=embeddings.device)
                embedding = embeddings[torch.arange(B), rand_idx].to(DEVICE)
                true_pose = true_poses[torch.arange(B), rand_idx].to(DEVICE).type(torch.float32)
                pred_pose = predictor(embedding)

                loss = F.mse_loss(pred_pose, true_pose)
                losses.update(loss.item()*perspective_range[1], embeddings[0].size(0))

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    print(colorstr('cyan',
                                   f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                   f'Training MSE-Loss {losses.avg:.4f}\t'))
                    print("=" * 60)
                batch += 1

            # ===== Validation =====
            predictor.eval()
            val_metrics = validate(predictor, val_loader, DEVICE, perspective_range)

            mlflow.log_metric('val_mse', val_metrics["mse"], step=epoch + 1)
            mlflow.log_metric('val_mae', val_metrics["mae"], step=epoch + 1)
            mlflow.log_metric('val_mae_pitch', val_metrics["mae_pitch"], step=epoch + 1)
            mlflow.log_metric('val_mae_yaw', val_metrics["mae_yaw"], step=epoch + 1)

            print(colorstr(
                'bright_green',
                f'Epoch {epoch + 1}/{NUM_EPOCH} | '
                f'Training MSE-Loss {losses.avg:.4f}\t'
                f'Validation MSE-Loss: {val_metrics["mse"]:.4f} | '
                f'Validation MAE: {val_metrics["mae"]:.4f} | '
                f'Validation Pitch MAE: {val_metrics["mae_pitch"]:.4f} | '
                f'Validation Yaw MAE: {val_metrics["mae_yaw"]:.4f}'
            ))
            #evaluate(model=predictor, data_loader=aflw_loader, device=DEVICE, perspective_range=perspective_range)
            print("#" * 60)

            time.sleep(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    main(cfg_yaml)