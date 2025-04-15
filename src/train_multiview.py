import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from src.aggregator.MeanAggregator import make_mean_aggregator
from src.aggregator.SEAggregator import make_se_aggregator
from src.aggregator.WeightedSumAggregator import make_weighted_sum_aggregator
from src.backbone.model_multiview_irse import IR_MV_50, execute_model
from src.util.Plotter import plot_weight_evolution
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.eval_model_multiview import evaluate_and_log_mvs
from src.util.load_checkpoint import load_checkpoint
from src.util.misc import colorstr
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, AverageMeter, accuracy

from torchinfo import summary
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse

if __name__ == '__main__':

    # ======= Read config =======#
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    TRAIN_SET = cfg['TRAIN_SET']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = cfg['AGG_NAME']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    TRAIN_ALL = cfg['TRAIN_ALL']  # Train whole Network
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']
    DISTANCE_METRIC = cfg['DISTANCE_METRIC']  # support: ['euclidian', 'cosine']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    use_face_corr = cfg['USE_FACE_CORR']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
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

    # ===== Test set naming ============
    test_bellus = 'test_rgb_bellus'
    test_bellus_fc = 'test_rgb_bellus_fc'
    test_facescape = 'test_rgb_facescape'
    test_faceverse = 'test_rgb_faceverse'
    test_texas = 'test_rgb_texas'
    test_bff = 'test_rgb_bff'
    test_bff_fc = 'test_rgb_bff_fc'
    test_vox2test = "test_vox2test"
    test_vox2train = "test_vox2train"

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
        log_dir = f'{LOG_ROOT}/tensorboard/{RUN_NAME}'

        train_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

        dataset_train = MultiviewDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=25, transform=train_transform, use_face_corr=use_face_corr)

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
        BACKBONE_DICT = {'IR_MV_50': IR_MV_50(INPUT_SIZE, EMBEDDING_SIZE)}
        BACKBONE_reg = BACKBONE_DICT[BACKBONE_NAME]
        BACKBONE_agg = BACKBONE_DICT[BACKBONE_NAME]
        model_stats = summary(BACKBONE_reg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
        print(colorstr('magenta', str(model_stats)))
        print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        print("=" * 60)

        AGG_DICT = {'WeightedSumAggregator': make_weighted_sum_aggregator([(25, 0), (26, 2), (26, 2), (26, 2), (26, 2)]),
                    'MeanAggregator': make_mean_aggregator([25, 25, 25, 25, 25]),
                    'SEAggregator': make_se_aggregator([64, 64, 128, 256, 512])}
        aggregators = AGG_DICT[AGG_NAME]

        HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID)}
        HEAD = HEAD_DICT[HEAD_NAME]
        print(colorstr('magenta', HEAD))
        print(colorstr('blue', f"{HEAD_NAME} Head Generated"))
        print("=" * 60)

        LOSS_DICT = {'Focal': FocalLoss(), 'Softmax': nn.CrossEntropyLoss()}
        LOSS = LOSS_DICT[LOSS_NAME]
        print(colorstr('magenta', LOSS))
        print(colorstr('blue', f"{LOSS_NAME} Loss Generated"))
        print("=" * 60)

        # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        if BACKBONE_NAME.find("IR") >= 0:
            backbone_paras_only_bn_reg, backbone_paras_wo_bn_reg = separate_irse_bn_paras(BACKBONE_reg)
            backbone_paras_only_bn_agg, backbone_paras_wo_bn_agg = separate_irse_bn_paras(BACKBONE_agg)
            _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
        else:
            backbone_paras_only_bn_reg, backbone_paras_wo_bn_reg = separate_resnet_bn_paras(BACKBONE_reg)
            backbone_paras_only_bn_agg, backbone_paras_wo_bn_agg = separate_resnet_bn_paras(BACKBONE_agg)
            _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

        if TRAIN_ALL:
            params_list = [{'params': backbone_paras_wo_bn_agg + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn_agg}].extend([{'params': i.parameters()} for i in aggregators])
        else:
            params_list = [{'params': head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}]
            params_list.extend([{'params': i.parameters()} for i in aggregators])

            for param in BACKBONE_reg.parameters():
                param.requires_grad = False
            for param in BACKBONE_agg.parameters():
                param.requires_grad = False

        OPTIMIZER_DICT = {'SGD': optim.SGD(params_list, lr=LR, momentum=MOMENTUM),
                          'ADAM': torch.optim.Adam(params_list, lr=LR)}

        OPTIMIZER = OPTIMIZER_DICT[OPTIMIZER_NAME]
        print(colorstr('magenta', OPTIMIZER))
        print(colorstr('blue', f"{OPTIMIZER_NAME} Optimizer Generated"))
        print("=" * 60)

        load_checkpoint(BACKBONE_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        load_checkpoint(BACKBONE_agg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        print("=" * 60)

        print(colorstr('magenta', f"Using face correspondences: {use_face_corr}"))
        print("=" * 60)

        # ======= GPU Settings =======
        if MULTI_GPU:
            BACKBONE_reg = nn.DataParallel(BACKBONE_reg, device_ids=GPU_ID)
            BACKBONE_agg = nn.DataParallel(BACKBONE_agg, device_ids=GPU_ID)
        BACKBONE_reg = BACKBONE_reg.to(DEVICE)
        BACKBONE_agg = BACKBONE_agg.to(DEVICE)
        for i in aggregators:
            i = i.to(DEVICE)
        # ======= train & validation & save checkpoint =======
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc # was 100
        NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
        NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
        batch = 0  # batch index
        # ======= Initialize early stopping parameters =======
        best_acc = 0  # Initial best value
        counter = 0  # Counter for epochs without improvement
        weights_log = [[] for _ in range(len(aggregators))]
        for epoch in range(NUM_EPOCH):
            # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plateau observed
            if epoch == STAGES[0]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[2]:
                schedule_lr(OPTIMIZER)

            #for i, agg in enumerate(aggregators):
            #    weights_log[i].append(agg.weights.detach().cpu().numpy().copy())

            #  ======= perform validation =======
            evaluate_and_log_mvs(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, test_bellus, epoch, (150, 150), BATCH_SIZE * 4, False, disable_bar=True)
            #evaluate_and_log_mvs(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, test_bff_fc, epoch, (150, 150), BATCH_SIZE * 4, use_face_corr, disable_bar=False)
            #evaluate_and_log_mvs(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, test_bff, epoch, (150, 150), BATCH_SIZE * 4, use_face_corr, disable_bar=False)
            evaluate_and_log_mvs(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_vox2testlc01", epoch, (112, 112), BATCH_SIZE * 4, use_face_corr, disable_bar=True)
            #evaluate_and_log_mvs(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, test_vox2train, epoch,(112, 112), BATCH_SIZE * 4, use_face_corr, disable_bar=False)
            print("=" * 60)

            BACKBONE_reg.eval()
            BACKBONE_agg.eval()
            [i.train() for i in aggregators]
            HEAD.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for inputs, labels, perspectives, face_corrs in tqdm(iter(train_loader)):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                        batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                if not use_face_corr and face_corrs.shape[1] > 0:
                    print("Using Feature Alignment")
                    use_face_corr = True

                labels = labels.to(DEVICE).long()
                _, embeddings = execute_model(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, inputs, perspectives, face_corrs, use_face_corr)
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

    plot_weight_evolution(weights_log, save_dir="weights_logs")
