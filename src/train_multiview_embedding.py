import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from src.aggregator.TransformerEmbeddingReducer import TransformerEmbeddingReducer
from src.util.datapipeline.MultiviewEmbeddingDataset import MultiviewEmbeddingDataset
from src.util.eval_model_multiview_embedding import evaluate_and_log_mv
from src.util.load_checkpoint import load_checkpoint
from src.util.misc import colorstr
from util.utils import make_weights_for_balanced_classes, separate_resnet_bn_paras, warm_up_lr, schedule_lr, AverageMeter, accuracy

from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse


def train(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where the datasets are stored
    TRAIN_SET = "rgb_bff_crop_emb"  # cfg['TRAIN_SET']
    LOG_ROOT = cfg['LOG_ROOT']
    BACKBONE_RESUME_ROOT = ""  # cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = "TransformerEmbeddingReducer"  # cfg['BACKBONE_NAME']

    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    ARC_S = cfg['ARC_S']
    ARC_M = cfg['ARC_M']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']

    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
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
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
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
        run_count = 0  # No runs if the experiment does not exist yet

    with mlflow.start_run(run_name=f"{RUN_NAME}_[{run_count + 1}]") as run:

        mlflow.log_param('config', cfg)
        print(f"{RUN_NAME}_{run_count + 1} ; run_id:", run.info.run_id)

        dataset_train = MultiviewEmbeddingDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=25)

        # create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.data, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS, drop_last=DROP_LAST
        )

        NUM_CLASS = len(train_loader.dataset.classes)
        print("Number of Training Classes: {}".format(NUM_CLASS))

        # ======= model & loss & optimizer =======
        BACKBONE = TransformerEmbeddingReducer(embedding_dim=512, num_heads=4, num_layers=1, dropout=0.1)
        model_stats_backbone = summary(BACKBONE, (BATCH_SIZE, NUM_VIEWS, 512), verbose=0)
        print(colorstr('magenta', str(model_stats_backbone)))
        print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with open(os.path.join(tmp_dir, 'Backbone_Summary.txt'), "w", encoding="utf-8") as f:
                f.write(str(model_stats_backbone))
            mlflow.log_artifacts(tmp_dir, artifact_path="ModelSummary")

        HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID, s=ARC_S, m=ARC_M),
                     'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID)}
        HEAD = HEAD_DICT[HEAD_NAME]
        print(colorstr('magenta', HEAD))
        print(colorstr('blue', f"{HEAD_NAME} Head Generated"))
        print("=" * 60)

        LOSS_DICT = {'Focal': FocalLoss(), 'Softmax': nn.CrossEntropyLoss()}
        LOSS = LOSS_DICT[LOSS_NAME]
        print(colorstr('blue', f"{LOSS_NAME} Loss Generated"))
        print("=" * 60)

        # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

        params_list = [{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}]

        OPTIMIZER_DICT = {'SGD': optim.SGD(params_list, lr=LR, momentum=MOMENTUM),
                          'ADAM': torch.optim.Adam(params_list, lr=LR)}

        OPTIMIZER = OPTIMIZER_DICT[OPTIMIZER_NAME]
        print(colorstr('magenta', OPTIMIZER))
        print("=" * 60)

        load_checkpoint(BACKBONE, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        print("=" * 60)

        # ======= GPU Settings =======
        if MULTI_GPU:
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)

        # ======= train & validation & save checkpoint =======
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc # was 100
        NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
        NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
        batch = 0  # batch index
        # ======= Initialize early stopping parameters =======
        best_acc = 0  # Initial best value
        counter = 0  # Counter for epochs without improvement
        for epoch in range(NUM_EPOCH):
            # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plateau observed
            if epoch == STAGES[0]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[2]:
                schedule_lr(OPTIMIZER)

            #  ======= perform validation =======
            bff_rr1 = evaluate_and_log_mv(DEVICE, BACKBONE, DATA_ROOT, "test_rgb_bff_crop_emb", epoch, BATCH_SIZE * 8, NUM_VIEWS, disable_bar=True)
            vox2test_rr1 = evaluate_and_log_mv(DEVICE, BACKBONE, DATA_ROOT, "test_vox2test_emb", epoch, BATCH_SIZE * 8, NUM_VIEWS, disable_bar=True)
            vox2train_rr1 = evaluate_and_log_mv(DEVICE, BACKBONE, DATA_ROOT, "test_vox2train_emb", epoch, BATCH_SIZE * 8, NUM_VIEWS, disable_bar=True)

            total_rr1 = (bff_rr1 + vox2test_rr1 + vox2train_rr1) / 3
            if epoch > 0 and best_rr1 < total_rr1:
                best_rr1 = total_rr1

            print("=" * 60)

            BACKBONE.train()
            HEAD.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for inputs, labels, _, _ in tqdm(iter(train_loader)):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                        batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                labels = labels.to(DEVICE).long()
                embeddings = BACKBONE(inputs)
                outputs = HEAD(embeddings, labels)
                loss = LOSS(outputs, labels)

                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs[0].size(0))
                top1.update(prec1.data.item(), inputs[0].size(0))
                top5.update(prec5.data.item(), inputs[0].size(0))

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                # display training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    print(colorstr('cyan',
                                   f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                   f'Training Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                   f'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                   f'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'))
                    print("=" * 60)
                batch += 1

            epoch_loss = losses.avg
            epoch_acc = top1.avg
            mlflow.log_metric('train_loss', epoch_loss, step=epoch + 1)
            mlflow.log_metric('Training_Accuracy', epoch_acc, step=epoch + 1)
            print("#" * 60)
            print(colorstr('bright_green', f'Epoch: {epoch + 1}/{NUM_EPOCH}\t'
                                           f'Training Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                           f'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                           f'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'))
            print("#" * 60)

            if epoch_acc > best_acc:  # Early stopping check
                best_acc = epoch_acc
                counter = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(colorstr('red', "=" * 60))
                    print(colorstr('red', " ======== Early stopping triggered ======== "))
                    print(colorstr('red', "=" * 60))
                    break

            time.sleep(0.2)
            # save checkpoints per epoch
            # if MULTI_GPU:
            #     torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            # else:
            #     torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))

    #plot_weight_evolution(weights_log, save_dir="weights_logs")
    return best_rr1


def training_objective(trial):

    cfg = cfg_yaml
    cfg['ARC_S'] = trial.suggest_float('ARC_S', 20, 100, log=True)
    cfg['ARC_M'] = trial.suggest_float('ARC_M', 0.1, 0.8, log=True)

    rr1 = train(cfg)
    return rr1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    #train(cfg_yaml)

    study = optuna.create_study(direction="maximize")
    study.optimize(training_objective, n_trials=100)

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    top_trials = study.best_trials[:5]
    for i, t in enumerate(top_trials):
        print(f"Trial {i + 1}: Accuracy={t.value}, Params={t.params}")
