import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from src.aggregator.Autoencoder import make_ae_head
from src.backbone.multiview_ires_lf import ir_mv_v2_18_lf, ir_mv_v2_34_lf, ir_mv_v2_50_lf, ir_mv_v2_100_lf, ir_mv_50_lf, \
    ir_mv_facenet_50_lf
from src.backbone.multiview_timmfr_lf import timm_mv_lf
from src.fuser.CosineDistanceWeightedAggregator import make_cosinedistance_fusion
from src.fuser.TransformerAggregator import make_transformer_fusion
from src.fuser.fuser import make_mlp_fusion, make_softmax_fusion
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.eval_model_multiview import evaluate_and_log_mv, evaluate_and_log_mv_verification
from src.util.load_checkpoint import load_checkpoint
from src.util.misc import colorstr
from util.utils import make_weights_for_balanced_classes, separate_bn_paras, warm_up_lr, schedule_lr, AverageMeter, train_accuracy
from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def eval_loop(backbone, data_root, epoch, batch_size, num_views, use_face_corr, eval_all, transform_size=(112, 112), final_crop=(112,112)):
    evaluate_and_log_mv(backbone, data_root, "test_rgb_bff_crop8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_vox2test_crop8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_vox2train_crop8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_nersemble8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_multipie_crop8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv(backbone, data_root, "test_multipie8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)
    evaluate_and_log_mv_verification(backbone, data_root, "test_ytf_crop8", epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']
    BACKBONE_RESUME_ROOT = os.path.join(os.getenv("BACKBONE_RESUME_ROOT"), cfg['BACKBONE_RESUME_PATH'])  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = cfg['AGG']['AGG_NAME']  # support: ['WeightedSumAggregator', 'MeanAggregator', 'SEAggregator']
    AGG_CONFIG = cfg['AGG']['AGG_CONFIG']  # Aggregator Config
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    HEAD_PARAMS = cfg.get('HEAD_PARAMS', [64.0, 0.50])
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']  # for normalize inputs
    use_face_corr = cfg['USE_FACE_CORR']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # embedding dimension
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    SHUFFLE_PERSPECTIVES = cfg.get('SHUFFLE_PERSPECTIVES', False)  # shuffle perspectives during train loop
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    PATIENCE = cfg.get('PATIENCE', 10)
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    WEIGHT_DECAY_HEAD = cfg.get('WEIGHT_DECAY_HEAD', 0.0005)
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate
    UNFREEZE_AGG_EPOCH = cfg.get('UNFREEZE_AGG_EPOCH', 1)  # Unfreeze aggregators after X Epochs. Train Arcface Head first for smoother fine-tuning
    STOPPING_CRITERION = cfg.get('STOPPING_CRITERION', 99.9)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    TORCH_COMPILE_MODE = cfg.get('TORCH_COMPILE_MODE', None)  # [None, default, reduce-overhead, max-autotune-no-cudagraphs, max-autotune]
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

        train_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            # transforms.RandomCrop((112, 112)),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

        dataset_train = MultiviewDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=NUM_VIEWS, transform=train_transform, use_face_corr=use_face_corr, shuffle_views=SHUFFLE_PERSPECTIVES)

        # create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.data, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS, drop_last=DROP_LAST
        )

        num_class = len(dataset_train.classes)
        print("Number of Training Classes: {}".format(num_class))

        # ======= Aggregator =======
        AGG_DICT = {'MLPFusion': lambda: make_mlp_fusion(),
                    'TransformerFusion': lambda: make_transformer_fusion(),
                    'SoftmaxFusion': lambda: make_softmax_fusion(),
                    'CosineDistanceAggregator': lambda: make_cosinedistance_fusion(NUM_VIEWS),
                    'AEFusion': lambda: make_ae_head(NUM_VIEWS, EMBEDDING_SIZE),
                    }
        aggregator = AGG_DICT[AGG_NAME]()
        model_arch = (BATCH_SIZE, NUM_VIEWS, 512)
        aggregator.to(DEVICE)
        model_stat = summary(aggregator, model_arch, verbose=0)
        print(colorstr('magenta', str(model_stat)))
        model_stats_agg = model_stat
        print("=" * 60)

        # ======= Backbone =======
        BACKBONE_DICT = {'IR_MV_Facenet_50': lambda: ir_mv_facenet_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'IR_MV_50': lambda: ir_mv_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'IR_MV_V2_18': lambda: ir_mv_v2_18_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'IR_MV_V2_34': lambda: ir_mv_v2_34_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'IR_MV_V2_50': lambda: ir_mv_v2_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'IR_MV_V2_100': lambda: ir_mv_v2_100_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                         'TIMM_MV': lambda: timm_mv_lf(DEVICE, aggregator)}

        BACKBONE = BACKBONE_DICT[BACKBONE_NAME]()
        BACKBONE.backbone_reg.to(DEVICE)
        model_stats_backbone = summary(BACKBONE.backbone_reg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
        print(colorstr('magenta', str(model_stats_backbone)))
        print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        print("=" * 60)

        # ======= HEAD & LOSS =======
        HEAD_DICT = {'ArcFace': lambda: ArcFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID, s=HEAD_PARAMS[0], m=HEAD_PARAMS[1]),
                     'CosFace': lambda: CosFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID),
                     'SphereFace': lambda: SphereFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID),
                     'Am_softmax': lambda: Am_softmax(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID)}
        HEAD = HEAD_DICT[HEAD_NAME]().to(DEVICE)

        #if TORCH_COMPILE_MODE:
        #    HEAD = torch.compile(HEAD, mode=TORCH_COMPILE_MODE)

        model_stats_head = summary(HEAD, input_size=[(BATCH_SIZE, EMBEDDING_SIZE), (BATCH_SIZE,)], dtypes=[torch.float, torch.long], verbose=0)
        print(colorstr('magenta', str(model_stats_head)))
        print(colorstr('blue', f"{HEAD_NAME} Head Generated"))
        print("=" * 60)

        LOSS_DICT = {'Focal': FocalLoss(), 'Softmax': nn.CrossEntropyLoss() }
        LOSS = LOSS_DICT[LOSS_NAME]
        print(colorstr('blue', f"{LOSS_NAME} Loss Generated"))
        print("=" * 60)

        # ======= Write Summaries =======
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with open(os.path.join(tmp_dir, 'Head_Summary.txt'), "w", encoding="utf-8") as f:
                f.write(str(model_stats_head))
            with open(os.path.join(tmp_dir, 'Backbone_Summary.txt'), "w", encoding="utf-8") as f:
                f.write(str(model_stats_backbone))
            with open(os.path.join(tmp_dir, 'Aggregator_Summary.txt'), "w", encoding="utf-8") as f:
                f.write(str(model_stats_agg))
            mlflow.log_artifacts(str(tmp_dir), artifact_path="ModelSummary")

        # ======= Optimizer Settings =======
        # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_bn_paras(HEAD)

        params_list = [{'params': head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY_HEAD}]
        OPTIMIZER_DICT = {'SGD': lambda: optim.SGD(params_list, lr=LR, momentum=MOMENTUM),
                          'ADAM': lambda: optim.Adam(params_list, lr=LR),
                          'ADAMW': lambda: optim.AdamW(params_list, lr=LR)}
        OPTIMIZER = OPTIMIZER_DICT[OPTIMIZER_NAME]()
        print(colorstr('magenta', OPTIMIZER))
        print("=" * 60)

        load_checkpoint(BACKBONE.backbone_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        print("=" * 60)

        # ======= Freezing Parameter Settings =======
        for param in aggregator.parameters():
            param.requires_grad = False
        for param in BACKBONE.backbone_reg.parameters():
            param.requires_grad = False

        print(colorstr('magenta', f"Using face correspondences: {use_face_corr}"))
        print(colorstr('magenta', f"Unfreezing aggregators at epoch: {UNFREEZE_AGG_EPOCH}"))
        print("=" * 60)

        # ======= Validation =======
        eval_loop(BACKBONE, DATA_ROOT, 0, BATCH_SIZE, NUM_VIEWS, use_face_corr, True, INPUT_SIZE, INPUT_SIZE)
        print("=" * 60)

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

            # =========== Gradient Handling ========
            if epoch == UNFREEZE_AGG_EPOCH:
                print(colorstr('yellow', f"Unfreezing aggregators at epoch {epoch}"))
                print(colorstr('yellow', "=" * 60))

                for param in aggregator.parameters():
                    param.requires_grad = True
                OPTIMIZER.add_param_group({'params': aggregator.parameters(), 'weight_decay': WEIGHT_DECAY})

            if epoch >= UNFREEZE_AGG_EPOCH:
                aggregator.train()
            else:
                aggregator.eval()

            BACKBONE.backbone_reg.eval()
            HEAD.train()

            # =========== Train Loop ========
            losses = AverageMeter()
            losses_arc = AverageMeter()
            losses_rec = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for step, (inputs, labels, perspectives, _, face_corrs, _) in enumerate(tqdm(iter(train_loader))):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                if not use_face_corr and face_corrs.shape[1] > 0:
                    print("Using Feature Alignment")
                    use_face_corr = True

                labels = labels.to(DEVICE).long()
                if AGG_NAME == "AEFusion":
                    lambda_rec = 1
                    aggregator.return_reconstruction = True
                    embeddings_reg, (embeddings_fused, embeddings_rec) = BACKBONE(inputs, perspectives, face_corrs, use_face_corr)
                    aggregator.return_reconstruction = False
                    outputs = HEAD(embeddings_fused, labels)
                    loss_arc = LOSS(outputs, labels)
                    embeddings_concat = torch.concatenate(embeddings_reg, dim=1)#torch.stack(embeddings_reg, dim=1)
                    loss_rec = F.mse_loss(embeddings_rec, embeddings_concat)
                    loss = loss_arc + lambda_rec * loss_rec

                    prec1, prec5 = train_accuracy(outputs.data, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs[0].size(0))
                    top1.update(prec1.item(), inputs[0].size(0))
                    top5.update(prec5.item(), inputs[0].size(0))

                    losses_arc.update(loss_arc.item(), inputs[0].size(0))
                    losses_rec.update(loss_rec.item(), inputs[0].size(0))

                else:
                    _, embeddings = BACKBONE(inputs, perspectives, face_corrs, use_face_corr)
                    outputs = HEAD(embeddings, labels)
                    loss = LOSS(outputs, labels)

                    prec1, prec5 = train_accuracy(outputs.data, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs[0].size(0))
                    top1.update(prec1.item(), inputs[0].size(0))
                    top5.update(prec5.item(), inputs[0].size(0))

                # mlflow.log_metric('train_batch_loss', losses.val, step=batch)

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    if AGG_NAME == "AEFusion":
                        print(colorstr('cyan',
                                       f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                       f'Training Loss {losses.avg:.4f} (arc: {losses_arc.avg:.5f} + rec: {losses_rec.avg:.5f})\t'
                                       f'Training Prec@1 {top1.avg:.3f}\t'
                                       f'Training Prec@5 {top5.avg:.3f}'))
                    else:
                        print(colorstr('cyan',
                                       f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                       f'Training Loss {losses.avg:.4f}\t'
                                       f'Training Prec@1 {top1.avg:.3f}\t'
                                       f'Training Prec@5 {top5.avg:.3f}'))
                    print("=" * 60)
                batch += 1

            mlflow.log_metric('train_loss', losses.avg, step=epoch + 1)
            mlflow.log_metric('Training_Accuracy', top1.avg, step=epoch + 1)
            print("#" * 60)
            if AGG_NAME == "AEFusion":
                print(colorstr('bright_green', f'Epoch: {epoch + 1}/{NUM_EPOCH}\t'
                                               f'Training Loss {losses.avg:.5f} (arc: {losses_arc.avg:.5f} + rec: {losses_rec.avg:.5f})\t'
                                               f'Training Prec@1 {top1.avg:.3f}\t'
                                               f'Training Prec@5 {top5.avg:.3f}'))
            else:
                print(colorstr('bright_green', f'Epoch: {epoch + 1}/{NUM_EPOCH}\t'
                                               f'Training Loss {losses.avg:.5f}\t'
                                               f'Training Prec@1 {top1.avg:.3f}\t'
                                               f'Training Prec@5 {top5.avg:.3f}'))
            print("#" * 60)

            #  ======= perform validation =======
            if epoch >= UNFREEZE_AGG_EPOCH:
                eval_loop(BACKBONE, DATA_ROOT, epoch, BATCH_SIZE, NUM_VIEWS, use_face_corr, False)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("=" * 60)

            if top1.avg > best_acc:  # Early stopping check
                best_acc = top1.avg
                counter = 0
            elif top1.avg > STOPPING_CRITERION and epoch > UNFREEZE_AGG_EPOCH:
                print(colorstr('red', "=" * 60))
                print(colorstr('red', f"======== Training Prec@1 reached > {STOPPING_CRITERION} -> Finishing ========"))
                print(colorstr('red', "=" * 60))
                break
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(colorstr('red', "=" * 60))
                    print(colorstr('red', " ======== Early stopping triggered ======== "))
                    print(colorstr('red', "=" * 60))
                    break

            time.sleep(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    main(cfg_yaml)
