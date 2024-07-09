import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from src.backbone.model_irse_rgbd import IR_152_rgbd, IR_101_rgbd, IR_50_rgbd, IR_SE_50_rgbd, IR_SE_101_rgbd, \
    IR_SE_152_rgbd
from src.backbone.model_resnet_rgbd import ResNet_50_rgbd, ResNet_101_rgbd, ResNet_152_rgbd
from src.util.ImageFolder4Channel import ImageFolder4Channel
from src.util.misc import colorstr
from util.eval_model import evaluate_and_log
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import mlflow
import yaml
import argparse

if __name__ == '__main__':

    # ======= Read config =======#

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config.yaml')
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
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    DISTANCE_METRIC = cfg['DISTANCE_METRIC']  # support: ['euclidian', 'cosine']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
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

    # ===== ML FLOW SET up ============
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
        writer = SummaryWriter(log_dir)

        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more online data augmentation
        train_transform = transforms.Compose([
            transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

        if 'rgbd' in TRAIN_SET:
            dataset_train = ImageFolder4Channel(os.path.join(DATA_ROOT, TRAIN_SET), train_transform)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, TRAIN_SET), train_transform)

        # create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS, drop_last=DROP_LAST
        )

        NUM_CLASS = len(train_loader.dataset.classes)
        print("Number of Training Classes: {}".format(NUM_CLASS))

        # ======= model & loss & optimizer =======#
        BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE, EMBEDDING_SIZE),
                         'ResNet_101': ResNet_101(INPUT_SIZE, EMBEDDING_SIZE),
                         'ResNet_152': ResNet_152(INPUT_SIZE, EMBEDDING_SIZE),
                         'ResNet_50_RGBD': ResNet_50_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'ResNet_101_RGBD': ResNet_101_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'ResNet_152_RGBD': ResNet_152_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_50': IR_50(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_101': IR_101(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_152': IR_152(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_50_RGBD': IR_50_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_101_RGBD': IR_101_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_152_RGBD': IR_152_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_50': IR_SE_50(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_101': IR_SE_101(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_152': IR_SE_152(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_50_RGBD': IR_SE_50_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_101_RGBD': IR_SE_101_rgbd(INPUT_SIZE, EMBEDDING_SIZE),
                         'IR_SE_152_RGBD': IR_SE_152_rgbd(INPUT_SIZE, EMBEDDING_SIZE)}
        if 'rgbd' in TRAIN_SET:
            BACKBONE_NAME = BACKBONE_NAME + '_RGBD'
        BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
        print("=" * 60)
        print(colorstr('magenta', BACKBONE))
        print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        print("=" * 60)

        # optionally resume from a checkpoint
        if BACKBONE_RESUME_ROOT:
            print("=" * 60)
            if os.path.isfile(BACKBONE_RESUME_ROOT):
                print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            else:
                print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))
            print("=" * 60)
        else:
            raise AttributeError('BACKBONE_RESUME_ROOT not activated')

        if MULTI_GPU:
            # multi-GPU setting
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
            BACKBONE = BACKBONE.to(DEVICE)
        else:
            # single-GPU setting
            BACKBONE = BACKBONE.to(DEVICE)

        #  ======= perform validation =======
        if 'rgbd' in TRAIN_SET:
            test_bellus = 'test_rgbd_bellus'
            test_facescape = 'test_rgbd_facescape'
            test_faceverse = 'test_rgbd_faceverse'
            test_texas = 'test_rgbd_texas'
        elif 'rgb' in TRAIN_SET:
            test_bellus = 'test_rgb_bellus'
            test_facescape = 'test_rgb_facescape'
            test_faceverse = 'test_rgb_faceverse'
            test_texas = 'test_rgb_texas'
        elif 'depth' in TRAIN_SET:
            test_bellus = 'test_depth_bellus'
            test_facescape = 'test_depth_facescape'
            test_faceverse = 'test_depth_faceverse'
            test_texas = 'test_depth_texas'

        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_bellus, writer, 0, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_facescape, writer, 0, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_faceverse, writer, 0, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_texas, writer, 0, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)

        print("=" * 60)
