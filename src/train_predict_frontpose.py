import tempfile
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.predictor.PoseFrontalizer import PoseFrontalizer, PoseFrontalizerWithPose
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


def validate(model, data_loader, device, with_pose):
    model.eval()

    cos_model_meter = AverageMeter()
    cos_baseline_meter = AverageMeter()

    with torch.no_grad():
        for embeddings, labels, scan_ids, true_poses, _, path in data_loader:
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
                model_input = torch.cat([input_emb, input_pose], dim=1)  # (B, D+2)
                pred_front = model(model_input)
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


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    LOG_ROOT = cfg['LOG_ROOT']

    PREDICTOR_NAME = cfg['PREDICTOR_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']

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
        #dataset_size = len(full_dataset)
        #split = int(0.9 * dataset_size)
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

        # ======= Predictor =======
        predictor_dict = {'PoseFrontalizer': lambda: PoseFrontalizer(),
                          'PoseFrontalizerWithPose': lambda : PoseFrontalizerWithPose()}

        predictor = predictor_dict[PREDICTOR_NAME]()
        predictor.to(DEVICE)
        if PREDICTOR_NAME == "PoseFrontalizerWithPose":
            model_stats_predictor = summary(predictor, (BATCH_SIZE, EMBEDDING_SIZE+2), verbose=0)
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

        # load_checkpoint(BACKBONE.backbone_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        # print("=" * 60)

        # ======= Validation =======
        print("#" * 60)
        val_metrics = validate(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")

        mlflow.log_metric('val_cosine_model', val_metrics["cosine_model"], step=0)
        mlflow.log_metric('val_cosine_baseline', val_metrics["cosine_baseline"], step=0)
        mlflow.log_metric('val_cosine_gain', val_metrics["cosine_gain"], step=0)

        print(colorstr(
            'bright_green',
            f'CosSim Model: {val_metrics["cosine_model"]:.4f} | '
            f'Baseline: {val_metrics["cosine_baseline"]:.4f} | '
            f'Gain: {val_metrics["cosine_gain"]:+.4f}'
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

            for step, (embeddings, labels, scan_ids, true_poses, _, path) in enumerate(tqdm(train_loader)):

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
                    model_input = torch.cat([input_emb, input_pose], dim=1)  # (B, D+2)
                    pred_front = predictor(model_input)
                else:
                    pred_front = predictor(input_emb)

                loss = cosine_loss(pred_front, front_emb)
                losses.update(loss.item(), B)

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
            val_metrics = validate(predictor, val_loader, DEVICE, PREDICTOR_NAME == "PoseFrontalizerWithPose")

            mlflow.log_metric('val_cosine_model', val_metrics["cosine_model"], step=epoch + 1)
            mlflow.log_metric('val_cosine_baseline', val_metrics["cosine_baseline"], step=epoch + 1)
            mlflow.log_metric('val_cosine_gain', val_metrics["cosine_gain"], step=epoch + 1)

            print(colorstr(
                'bright_green',
                f'Epoch {epoch + 1}/{NUM_EPOCH} | '
                f'Train CosLoss: {losses.avg:.4f} | '
                f'Val CosSim: {val_metrics["cosine_model"]:.4f} | '
                f'Baseline: {val_metrics["cosine_baseline"]:.4f} | '
                f'Gain: {val_metrics["cosine_gain"]:+.4f}'
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
