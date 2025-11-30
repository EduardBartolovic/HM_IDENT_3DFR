import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from loss.focal import FocalLoss
from src.predictor.EmbeddingHPE import PoseAndFrontalizer
from src.predictor.PoseFrontalizer import PoseFrontalizer
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.eval_model_multiview import evaluate_and_log_mv, evaluate_and_log_mv_verification
from src.util.load_checkpoint import load_checkpoint
from src.util.misc import colorstr
from util.utils import make_weights_for_balanced_classes, separate_bn_paras, warm_up_lr, schedule_lr, AverageMeter, \
    train_accuracy
from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def eval_loop(backbone, data_root, epoch, batch_size, num_views, use_face_corr, eval_all, transform_size=(112, 112),
              final_crop=(112, 112)):
    evaluate_and_log_mv(backbone, data_root, "test_rgb_bff_crop8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_vox2test_crop8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_vox2train_crop8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_nersemble8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_multipie_crop8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_multipie8", epoch, transform_size, final_crop, batch_size * 4,
                        num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv_verification(backbone, data_root, "test_ytf_crop8", epoch, transform_size, final_crop,
                                     batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"),
                             cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']

    PREDICTOR_NAME = cfg[
        'PREDICTOR_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

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
    STOPPING_CRITERION = cfg.get('STOPPING_CRITERION', 99.9)

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

        # create a weighted random sampler to process imbalanced data
        # weights = make_weights_for_balanced_classes(dataset_train.samples, len(dataset_train.classes))
        # weights = torch.DoubleTensor(weights)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [split, dataset_size - split],
            generator=torch.Generator().manual_seed(SEED)
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
        predictor_dict = {'PoseAndFrontalizer': lambda: PoseAndFrontalizer(),
                          "PoseFrontalizer": lambda: PoseFrontalizer()}

        predictor = predictor_dict[PREDICTOR_NAME]()
        predictor.to(DEVICE)
        model_stats_predictor = summary(predictor, (torch.randn(BATCH_SIZE, EMBEDDING_SIZE).to(DEVICE), torch.randn(BATCH_SIZE, 2).to(DEVICE)), verbose=0)
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
        # eval_loop(predictor, DATA_ROOT, 0, BATCH_SIZE, NUM_VIEWS, use_face_corr, True, INPUT_SIZE, INPUT_SIZE)
        # print("=" * 60)

        # ======= train & early stopping parameters=======
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc
        NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
        NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
        batch = 0  # batch index
        best_acc = 0  # Initial best value
        counter = 0  # Counter for epochs without improvement
        for epoch in range(0, NUM_EPOCH):

            if epoch in STAGES:
                schedule_lr(OPTIMIZER)

            predictor.train()

            # =========== Train Loop ========
            losses = AverageMeter()
            losses_hpe = AverageMeter()
            losses_posenorm = AverageMeter()
            default_pose_errors = AverageMeter()
            for step, (embeddings, labels, scan_ids, ref_p, true_poses, path) in enumerate(tqdm(iter(train_loader))):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                B, V, D = embeddings.shape
                rand_idx = torch.randint(0, V, (B,), device=embeddings.device)

                embedding = embeddings[torch.arange(B), rand_idx].to(DEVICE)

                frontal_idx = (true_poses[0] == torch.tensor([0, 0])).all(dim=1).nonzero(as_tuple=True)[0]
                emb_frontal = embeddings[torch.arange(B), frontal_idx].to(DEVICE)

                true_pose = true_poses[torch.arange(B), rand_idx].to(DEVICE)

                #pred_pose, pred_emb_front = predictor(embedding)
                pred_emb_front = predictor(embedding, true_pose)

                #loss_hpe = F.smooth_l1_loss(pred_pose, true_pose)

                pred_norm = F.normalize(pred_emb_front, dim=1)
                front_norm = F.normalize(emb_frontal, dim=1)
                loss_emb = 1 - F.cosine_similarity(pred_norm, front_norm, dim=1).mean()
                default_pose_error = 1 - F.cosine_similarity(embedding, front_norm, dim=1).mean()

                #loss_emb = F.mse_loss(pred_emb_front, emb_frontal)

                #loss = loss_hpe + loss_emb
                loss = loss_emb

                losses.update(loss.item(), embeddings[0].size(0))

                #losses_hpe.update(loss_hpe.item(), embeddings[0].size(0))
                losses_posenorm.update(loss_emb.item(), embeddings[0].size(0))
                default_pose_errors.update(default_pose_error.item(), embeddings[0].size(0))

                # mlflow.log_metric('train_batch_loss', losses.val, step=batch)

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    print(colorstr('cyan',
                                   f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                   f'Training Loss {losses.avg:.4f}\t'
                                   f'Loss-HPE {losses_hpe.avg:.3f}\t'
                                   f'Loss-PoseNorm {losses_posenorm.avg:.3f}\t'
                                   f'Default-PoseError {default_pose_errors.avg:.3f}'))
                    print("=" * 60)
                batch += 1

            # ===== Validation =====
            predictor.eval()
            val_losses = AverageMeter()
            val_default_pose_errors = AverageMeter()
            with torch.no_grad():
                for embeddings, labels, scan_ids, ref_p, true_poses, path in val_loader:
                    B, V, D = embeddings.shape
                    rand_idx = torch.randint(0, V, (B,), device=embeddings.device)

                    embedding = embeddings[torch.arange(B), rand_idx].to(DEVICE)

                    frontal_idx = (true_poses[0] == torch.tensor([0, 0])).all(dim=1).nonzero(as_tuple=True)[0]
                    emb_frontal = embeddings[torch.arange(B), frontal_idx].to(DEVICE)

                    # TODO: Learn not front but specific Random pose
                    true_pose = true_poses[torch.arange(B), rand_idx].to(DEVICE)

                    pred_emb_front = predictor(embedding, true_pose)

                    pred_norm = F.normalize(pred_emb_front, dim=1)
                    front_norm = F.normalize(emb_frontal, dim=1)
                    loss_emb = 1 - F.cosine_similarity(pred_norm, front_norm, dim=1).mean()

                    default_pose_error_val = 1 - F.cosine_similarity(embedding, front_norm, dim=1).mean()
                    val_default_pose_errors.update(default_pose_error_val.item())
                    val_losses.update(loss_emb.item())

            mlflow.log_metric('train_loss', losses.avg, step=epoch + 1)
            print("#" * 60)
            print(colorstr('bright_green', f'Epoch: {epoch + 1}/{NUM_EPOCH}\t'
                                           f'Training Loss {losses.avg:.4f}\t'
                                           #f'Loss-HPE {losses_hpe.avg:.3f}\t'
                                           #f'Loss-PoseNorm {losses_posenorm.avg:.3f}\t'
                                           f'Train Default-PoseError {default_pose_errors.avg:.3f}\t'
                                           f'Val Loss: {val_losses.avg:.4f}\t'
                                           f'Val Default-PoseError {val_default_pose_errors.avg:.3f}\t'))
            print("#" * 60)



            #if top1.avg > best_acc:  # Early stopping check
            #    best_acc = top1.avg
            #    counter = 0
            #elif top1.avg > STOPPING_CRITERION and epoch > UNFREEZE_AGG_EPOCH:
            #    print(colorstr('red', "=" * 60))
            #    print(colorstr('red', f"======== Training Prec@1 reached > {STOPPING_CRITERION} -> Finishing ========"))
            #    print(colorstr('red', "=" * 60))
            #    break
            #else:
            #    counter += 1
            #    if counter >= PATIENCE:
            #        print(colorstr('red', "=" * 60))
            #        print(colorstr('red', " ======== Early stopping triggered ======== "))
            #        print(colorstr('red', "=" * 60))
            #        break

            time.sleep(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    main(cfg_yaml)
