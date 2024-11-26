import torch
import torch.nn as nn
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from src.backbone.model_irse_rgbd import IR_152_rgbd, IR_101_rgbd, IR_50_rgbd, IR_SE_50_rgbd, IR_SE_101_rgbd, \
    IR_SE_152_rgbd
from src.backbone.model_resnet_rgbd import ResNet_50_rgbd, ResNet_101_rgbd, ResNet_152_rgbd
from src.util.eval_model_verification import evaluate_verification_lfw, evaluate_verification_colorferet
from src.util.misc import colorstr
from util.eval_model import evaluate_and_log

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

    BACKBONE_NAME = cfg[
        'BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD')  # support: ['SGD', 'ADAM']
    DISTANCE_METRIC = cfg['DISTANCE_METRIC']  # support: ['euclidian', 'cosine']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = 0
    PATIENCE = 0
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

        if BACKBONE_RESUME_ROOT:
            print("=" * 60)
            if os.path.isfile(BACKBONE_RESUME_ROOT):
                print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, weights_only=True))
            else:
                print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))
            print("=" * 60)
        else:
            raise AttributeError('BACKBONE_RESUME_ROOT not activated')

        if MULTI_GPU:
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
            BACKBONE = BACKBONE.to(DEVICE)
        else:
            BACKBONE = BACKBONE.to(DEVICE)

        #  ======= perform validation =======
        if 'rgbd' in TRAIN_SET:
            test_bellus = 'test_rgbd_bellus'
            test_facescape = 'test_rgbd_facescape'
            test_faceverse = 'test_rgbd_faceverse'
            test_texas = 'test_rgbd_texas'
            test_bff = 'test_rgbd_bff'
        elif 'rgb' in TRAIN_SET or 'photo' in TRAIN_SET:
            test_bellus = 'test_rgb_bellus'
            test_facescape = 'test_rgb_facescape'
            test_faceverse = 'test_rgb_faceverse'
            test_texas = 'test_rgb_texas'
            test_bff = 'test_rgb_bff'
        elif 'depth' in TRAIN_SET:
            test_bellus = 'test_depth_bellus'
            test_facescape = 'test_depth_facescape'
            test_faceverse = 'test_depth_faceverse'
            test_texas = 'test_depth_texas'
            test_bff = 'test_depth_bff'

        if 'rgbd' not in TRAIN_SET:
            # evaluate_verification_lfw(DEVICE, BACKBONE, DATA_ROOT, 'test_lfw_deepfunneled', 0, DISTANCE_METRIC, test_transform, BATCH_SIZE)
            # evaluate_verification_colorferet(DEVICE, BACKBONE, DATA_ROOT, 'test_colorferet', 0, DISTANCE_METRIC, test_transform, BATCH_SIZE)
            # print(colorstr('blue', "=" * 60))
            evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, 'test_photo_bellus', 0, DISTANCE_METRIC, (200, 150), BATCH_SIZE)
            evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, 'test_photo_colorferet1_n', 0, DISTANCE_METRIC, (150, 150), BATCH_SIZE)

        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_bellus, 0, DISTANCE_METRIC, (150, 150), BATCH_SIZE)
        # evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_facescape, 0, DISTANCE_METRIC, (112, 112), BATCH_SIZE)
        # evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_faceverse, 0, DISTANCE_METRIC, (112, 112), BATCH_SIZE)
        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_texas, 0, DISTANCE_METRIC, (168, 112), BATCH_SIZE)
        evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_bff, 0, DISTANCE_METRIC, (150, 150), BATCH_SIZE)

        print("=" * 60)
