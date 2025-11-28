import numpy as np
import torch
import torchvision.transforms as transforms

from src.aggregator.MeanAggregator import make_mean_aggregator
from src.backbone.multiview_ires import ir_mv_v2_50, ir_mv_v2_34, ir_mv_v2_18, ir_mv_v2_100, ir_mv_50, ir_mv_facenet
from src.backbone.multiview_timmfr import timm_mv
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.load_checkpoint import load_checkpoint
from src.util.misc import colorstr
from torchinfo import summary
from tqdm import tqdm
import os
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)
    DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), cfg['DATA_ROOT_PATH'])  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    BACKBONE_RESUME_ROOT = os.path.join(os.getenv("BACKBONE_RESUME_ROOT"), cfg['BACKBONE_RESUME_PATH'])  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = cfg['AGG']['AGG_NAME']  # support: ['WeightedSumAggregator', 'MeanAggregator', 'SEAggregator']
    AGG_CONFIG = cfg['AGG']['AGG_CONFIG']  # Aggregator Config

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']  # for normalize inputs
    use_face_corr = cfg['USE_FACE_CORR']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # embedding dimension
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    SHUFFLE_PERSPECTIVES = cfg.get('SHUFFLE_PERSPECTIVES', False)  # shuffle perspectives during train loop

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    dataset_train = MultiviewDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=NUM_VIEWS, transform=train_transform, use_face_corr=use_face_corr, shuffle_views=SHUFFLE_PERSPECTIVES)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST
    )

    # ======= Aggregator =======
    agg_dict = {'MeanAggregator': lambda: make_mean_aggregator(AGG_CONFIG)}
    aggregators = agg_dict[AGG_NAME]()

    # ======= Backbone =======
    BACKBONE_DICT = {'IR_MV_Facenet': lambda: ir_mv_facenet(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_MV_50': lambda: ir_mv_50(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_MV_V2_18': lambda: ir_mv_v2_18(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_MV_V2_34': lambda: ir_mv_v2_34(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_MV_V2_50': lambda: ir_mv_v2_50(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_MV_V2_100': lambda: ir_mv_v2_100(DEVICE, aggregators, EMBEDDING_SIZE),
                     'TIMM_MV': lambda: timm_mv(DEVICE, aggregators, EMBEDDING_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]()
    BACKBONE.backbone_reg.to(DEVICE)
    BACKBONE.backbone_agg.to(DEVICE)
    model_stats_backbone = summary(BACKBONE.backbone_agg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
    print(colorstr('magenta', str(model_stats_backbone)))
    print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
    print("=" * 60)

    load_checkpoint(BACKBONE.backbone_reg, None, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
    load_checkpoint(BACKBONE.backbone_agg, None, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd='rgbd' in TRAIN_SET)
    print("=" * 60)

    # ======= Freezing Parameter Settings =======
    for agg in aggregators:
        for param in agg.parameters():
            param.requires_grad = False
    for param in BACKBONE.backbone_agg.parameters():
        param.requires_grad = False
    for param in BACKBONE.backbone_reg.parameters():
        param.requires_grad = False

    [agg.eval() for agg in aggregators]
    BACKBONE.backbone_agg.eval()
    BACKBONE.backbone_reg.eval()

    for step, (inputs, class_idx, ref_perspectives, true_perspectives, facial_corr, scan_id) in enumerate(tqdm(iter(train_loader))):

        embeddings_reg, _ = BACKBONE(inputs, ref_perspectives, facial_corr, use_face_corr)

        embeddings_reg_list = []
        for emb in embeddings_reg:
            embeddings_reg_list.append(emb.detach().cpu().numpy())
        emb_reg_np = np.array(embeddings_reg_list).transpose(1, 0, 2)
        class_idx_np = np.array(class_idx)
        scan_id_np = np.array(scan_id)

        ref_perspectives_np = np.array(ref_perspectives).transpose(1, 0)
        true_perspectives_np = np.array(true_perspectives).transpose(1, 0)

        for i in range(BATCH_SIZE-1):
            sample_name = f"{scan_id[i]}.npz"
            save_dir = os.path.join("E:\\Download\\dataset\\", str(class_idx_np[i]))
            sample_path = os.path.join(save_dir, sample_name)

            os.makedirs(save_dir, exist_ok=True)
            np.savez(
                sample_path,
                embedding_reg=emb_reg_np[i],
                label=class_idx_np[i],
                scan_id=scan_id_np[i],
                ref_perspective=ref_perspectives_np[i],
                true_perspective=true_perspectives_np[i],
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)
    main(cfg_yaml)
