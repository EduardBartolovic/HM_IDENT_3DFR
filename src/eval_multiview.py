import random

import torch
import torch.nn as nn

from src.aggregator.MeanAggregator import make_mean_aggregator
from src.aggregator.SEAggregator import make_se_aggregator
from src.aggregator.WeightedSumAggregator import make_weighted_sum_aggregator
from src.backbone.model_multiview_irse import IR_MV_50
from src.util.eval_model_multiview import evaluate_and_log_mv
from src.util.load_checkpoint import load_checkpoint
import os
import mlflow
import yaml
import argparse


def main(cfg, test):
    # ======= Read config =======#
    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    TRAIN_SET = cfg['TRAIN_SET']
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = cfg['AGG']['AGG_NAME']
    AGG_CONFIG = cfg['AGG']['AGG_CONFIG']

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    use_face_corr = cfg['USE_FACE_CORR']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    #print("=" * 60)
    #print("Overall Configurations:")
    #print(cfg)
    #print("=" * 60)

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

        # ======= model & loss & optimizer =======
        BACKBONE_DICT = {'IR_MV_50': IR_MV_50(INPUT_SIZE, EMBEDDING_SIZE)}
        BACKBONE_reg = BACKBONE_DICT[BACKBONE_NAME]
        BACKBONE_agg = BACKBONE_DICT[BACKBONE_NAME]
        #model_stats = summary(BACKBONE_reg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
        #print(colorstr('magenta', str(model_stats)))
        #print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        #print("=" * 60)

        AGG_DICT = {'WeightedSumAggregator': make_weighted_sum_aggregator(AGG_CONFIG),
                    'MeanAggregator': make_mean_aggregator([25, 25, 25, 25, 25]),
                    'SEAggregator': make_se_aggregator([64, 64, 128, 256, 512])}
        aggregators = AGG_DICT[AGG_NAME]

        #print(colorstr('magenta', HEAD))
        #print(colorstr('blue', f"{HEAD_NAME} Head Generated"))
        #print("=" * 60)

        load_checkpoint(BACKBONE_reg, None, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        load_checkpoint(BACKBONE_agg, None, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
        #print("=" * 60)

        #print(colorstr('magenta', f"Using face correspondences: {use_face_corr}"))
        #print("=" * 60)

        # ======= GPU Settings =======
        if MULTI_GPU:
            BACKBONE_reg = nn.DataParallel(BACKBONE_reg, device_ids=GPU_ID)
            BACKBONE_agg = nn.DataParallel(BACKBONE_agg, device_ids=GPU_ID)
        BACKBONE_reg = BACKBONE_reg.to(DEVICE)
        BACKBONE_agg = BACKBONE_agg.to(DEVICE)
        for agg in aggregators:
            agg.to(DEVICE)

        #  ======= perform validation =======

        evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, test, 0, (112, 112), BATCH_SIZE * 4, NUM_VIEWS, use_face_corr, disable_bar=True)

        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_rgb_bellus_crop", 0, (112, 112), BATCH_SIZE * 4, NUM_VIEWS, False, disable_bar=True)
        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_rgb_bff_crop", 0, (112, 112), BATCH_SIZE * 4, NUM_VIEWS, use_face_corr, disable_bar=True)
        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_rgb_bff", 0, (150, 150), BATCH_SIZE * 4, NUM_VIEWS, use_face_corr, disable_bar=False)
        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_vox2test", 0, (112, 112), BATCH_SIZE * 4, NUM_VIEWS, use_face_corr, disable_bar=True)
        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_nersemble", 0, (112, 112),BATCH_SIZE * 4, use_face_corr, disable_bar=False)
        #evaluate_and_log_mv(DEVICE, BACKBONE_reg, BACKBONE_agg, aggregators, DATA_ROOT, "test_vox2train", 0,(112, 112), BATCH_SIZE * 4, use_face_corr, disable_bar=False)
        print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        yaml_cfg = yaml.safe_load(file)

    mass_dataset_eval = True
    mass_config_test = False

    if mass_dataset_eval:
        for i in os.listdir(yaml_cfg['DATA_ROOT']):
            if "test_" in i:
                views = i.count("_")-4
                print("################ RUNNING: ", i, "with views: ", views)
                yaml_cfg["AGG"]["AGG_CONFIG"] = [[views, 0], [views+1, 0], [views+1, 0], [views+1, 0], [views+1, 0]]
                yaml_cfg['NUM_VIEWS'] = views
                main(yaml_cfg, i)

    if mass_config_test:
        for i in range(0, 100):
            views = 25
            yaml_cfg["AGG"]["AGG_CONFIG"] = [
                [views, random.randint(0, 5)],
                [views + 1, random.randint(0, 5)],
                [views + 1, random.randint(0, 5)],
                [views + 1, random.randint(0, 5)],
                [views + 1, random.randint(0, 5)],
            ]
            yaml_cfg['NUM_VIEWS'] = views
            print("################ RUNNING: ", yaml_cfg["AGG"]["AGG_CONFIG"])
            main(yaml_cfg, i)
