

import torch

from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from src.aggregator.ConvAggregator import make_conv_aggregator
from src.aggregator.CosineDistanceWeightedAggregator import make_cosinedistance_weighted_aggregator
from src.aggregator.EmbeddingWeightedAggregator import make_embeddingweighted_aggregator
from src.aggregator.MaxAggregator import make_max_aggregator
from src.aggregator.MeanAggregator import make_mean_aggregator
from src.aggregator.MedianAggregator import make_median_aggregator
from src.aggregator.RobustMeanAggregator import make_rma
from src.aggregator.SEAggregator import make_se_aggregator
from src.aggregator.TransformerAggregator import make_transformer_aggregator
from src.aggregator.TransformerAggregatorV2 import make_transformerv2_aggregator
from src.aggregator.WeightedSumAggregator import make_weighted_sum_aggregator
from src.backbone.multiview_ires import ir_mv_v2_50, ir_mv_v2_34, ir_mv_v2_18, ir_mv_v2_100, ir_mv_50, ir_mv_facenet
from src.backbone.multiview_timmfr import timm_mv
from src.util.eval_model_multiview import evaluate_and_log_mv
from src.util.load_checkpoint import load_checkpoint
import os
import mlflow
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def eval_loop(backbone, data_root, epoch, batch_size, num_views, use_face_corr, eval_all, TEST_SET, transform_size=(112, 112), final_crop=(112, 112)):
    evaluate_and_log_mv(backbone, data_root, TEST_SET, epoch, transform_size, final_crop, batch_size * 4, num_views, use_face_corr, disable_bar=True, eval_all=eval_all)


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = "/home/gustav/dataset11-bff/"  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    LOG_ROOT = cfg['LOG_ROOT']
    BACKBONE_RESUME_ROOT = os.path.join(os.getenv("BACKBONE_RESUME_ROOT"), cfg['BACKBONE_RESUME_PATH'])  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = cfg['AGG']['AGG_NAME']  # support: ['WeightedSumAggregator', 'MeanAggregator', 'SEAggregator']
    AGG_CONFIG = cfg['AGG']['AGG_CONFIG']  # Aggregator Config
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    HEAD_PARAMS = cfg.get('HEAD_PARAMS', [64.0, 0.50])

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']  # for normalize inputs
    use_face_corr = cfg['USE_FACE_CORR']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # embedding dimension
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids

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

        # ======= Aggregator =======
        agg_dict = {'WeightedSumAggregator': lambda: make_weighted_sum_aggregator(AGG_CONFIG),
                    'MeanAggregator': lambda: make_mean_aggregator(AGG_CONFIG),
                    'RobustMeanAggregator': lambda: make_rma([NUM_VIEWS]*5),
                    'MedianAggregator': lambda: make_median_aggregator([NUM_VIEWS]*5),
                    'MaxAggregator': lambda: make_max_aggregator(AGG_CONFIG),
                    'ConvAggregator': lambda: make_conv_aggregator(AGG_CONFIG),
                    'SEAggregator': lambda: make_se_aggregator([64, 64, 128, 256, 512]),
                    'TransformerAggregator': lambda: make_transformer_aggregator([64, 64, 128, 256, 512], NUM_VIEWS, AGG_CONFIG),
                    'TransformerAggregatorV2': lambda: make_transformerv2_aggregator([64, 64, 128, 256, 512], NUM_VIEWS, AGG_CONFIG),
                    'EmbeddingWeightedAggregator': lambda: make_embeddingweighted_aggregator(AGG_CONFIG),
                    'CosineDistanceAggregator': lambda: make_cosinedistance_weighted_aggregator(AGG_CONFIG),}
        aggregators = agg_dict[AGG_NAME]()
        model_arch = [(BATCH_SIZE, NUM_VIEWS, 64, 112, 112), (BATCH_SIZE, NUM_VIEWS, 64, 56, 56), (BATCH_SIZE, NUM_VIEWS, 128, 28, 28), (BATCH_SIZE, NUM_VIEWS+1, 256, 14, 14), (BATCH_SIZE, NUM_VIEWS+1, 512, 7, 7)]
        for agg, arch in zip(aggregators, model_arch):
            agg.to(DEVICE)

        # ======= Backbone =======
        BACKBONE_DICT = {'IR_MV_Facenet': lambda: ir_mv_facenet(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'IR_MV_50': lambda: ir_mv_50(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'IR_MV_V2_18': lambda: ir_mv_v2_18(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'IR_MV_V2_34': lambda: ir_mv_v2_34(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'IR_MV_V2_50': lambda: ir_mv_v2_50(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'IR_MV_V2_100': lambda: ir_mv_v2_100(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5}),
                         'TIMM_MV': lambda: timm_mv(DEVICE, aggregators, EMBEDDING_SIZE, active_stages={2, 3, 4, 5})}
        BACKBONE = BACKBONE_DICT[BACKBONE_NAME]()
        BACKBONE.backbone_reg.to(DEVICE)
        BACKBONE.backbone_agg.to(DEVICE)

        # ======= HEAD & LOSS =======
        num_class = 100
        HEAD_DICT = {'ArcFace': lambda: ArcFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID, s=HEAD_PARAMS[0], m=HEAD_PARAMS[1]),
                     'CosFace': lambda: CosFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID),
                     'SphereFace': lambda: SphereFace(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID),
                     'Am_softmax': lambda: Am_softmax(in_features=EMBEDDING_SIZE, out_features=num_class, device_id=GPU_ID)}
        HEAD = HEAD_DICT[HEAD_NAME]().to(DEVICE)

        load_checkpoint(BACKBONE.backbone_reg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT)
        load_checkpoint(BACKBONE.backbone_agg, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT)

        # ======= Validation =======
        eval_loop(BACKBONE, DATA_ROOT, 0, BATCH_SIZE, NUM_VIEWS, use_face_corr, True, cfg_copy['TEST_SET'], INPUT_SIZE, INPUT_SIZE)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)

    # Resolve root folder
    base_data_root = "/home/gustav/dataset11-bff/"

    # find all subdirectories
    subfolders = [f for f in os.listdir(base_data_root) if os.path.isdir(os.path.join(base_data_root, f))]

    if not subfolders:
        print(f"No subfolders found in {base_data_root}. Exiting.")
        exit()

    print(f"Found {len(subfolders)} datasets: {subfolders}")
    print("=" * 80)

    # Loop through each folder and run main(cfg) for each
    for folder in subfolders:
        print(f"\n🚀 Running for dataset: {folder}")

        num_views_cfg = len(folder.replace("test_rgb_bff_crop_new_", "").split())

        cfg_copy = dict(cfg_yaml)
        cfg_copy['TEST_SET'] = folder
        cfg_copy['NUM_VIEWS'] = num_views_cfg

        main(cfg_copy)

    print("\n✅ All dataset runs finished.")
