import numpy as np
import torch
import torchvision.transforms as transforms

from src.aggregator.MeanAggregator import make_mean_aggregator
from src.backbone.multiview_ires import ir_mv_v2_50, ir_mv_v2_34, ir_mv_v2_18, ir_mv_v2_100, ir_mv_50, ir_mv_facenet
from src.backbone.multiview_onnx import onnx_mv
from src.backbone.multiview_timmfr import timm_mv
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.load_checkpoint import load_checkpoint
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def main(cfg):
    SEED = 42
    torch.manual_seed(SEED)
    DATA_ROOT = cfg['DATA_ROOT_PATH']  # the parent root where the datasets are stored
    TRAIN_SET = cfg['TRAIN_SET']
    BACKBONE_RESUME_ROOT = os.path.join(os.getenv("BACKBONE_RESUME_ROOT"), cfg['BACKBONE_RESUME_PATH'])  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    AGG_NAME = "MeanAggregator"  # support: ['WeightedSumAggregator', 'MeanAggregator', 'SEAggregator']

    AGG_CONFIG = {'ACTIVE_STAGES': [False, False, False, False, False]}

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
    RGB_MEAN = [0.5, 0.5, 0.5]   # for normalize inputs
    RGB_STD = [0.5, 0.5, 0.5]   # for normalize inputs
    use_face_corr = False
    EMBEDDING_SIZE = 512  # embedding dimension
    BATCH_SIZE = 16  # Batch size in training
    DROP_LAST = False

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = True
    NUM_WORKERS = 8
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

    dataset_train = MultiviewDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=NUM_VIEWS, transform=train_transform, use_face_corr=use_face_corr, shuffle_views=False)

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
                     'TIMM_MV': lambda: timm_mv(DEVICE, aggregators, EMBEDDING_SIZE),
                     'ONNX_MV': lambda: onnx_mv(DEVICE, BACKBONE_RESUME_ROOT)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]()
    BACKBONE.backbone_reg.to(DEVICE)
    BACKBONE.backbone_agg.to(DEVICE)
    #model_stats_backbone = summary(BACKBONE.backbone_reg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
    #print(colorstr('magenta', str(model_stats_backbone)))
    #print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
    #print("=" * 60)

    load_checkpoint(BACKBONE.backbone_reg, None, BACKBONE_RESUME_ROOT, "", rgbd='rgbd' in TRAIN_SET)
    #load_checkpoint(BACKBONE.backbone_agg, None, BACKBONE_RESUME_ROOT, "", rgbd='rgbd' in TRAIN_SET)
    print("=" * 60)

    BACKBONE.eval()

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

        for i in range(min(BATCH_SIZE-1, len(scan_id))):
            sample_name = f"{scan_id[i]}.npz"
            save_dir = os.path.join(cfg['OUT'], str(class_idx_np[i]))
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
    cfg_yaml = {}
    cfg_yaml['DATA_ROOT_PATH'] = "F:\\Face\\data\\dataset13\\"
    cfg_yaml["TRAIN_SET"] = "rgb_bff_crop187"
    cfg_yaml['BACKBONE_RESUME_PATH'] = "F:\\Face\\HM_IDENT_3DFR\\pretrained\\glint_cosface_r18_fp16.pth"
    cfg_yaml['BACKBONE_NAME'] = "IR_MV_V2_18" #"ONNX_MV"

    cfg_yaml['INPUT_SIZE'] = [112, 112]
    cfg_yaml['NUM_VIEWS'] = 187  # Number of views

    cfg_yaml['OUT'] = "F:\\Face\\data\\dataset13_emb\\" + cfg_yaml["TRAIN_SET"] + "_emb-irseglintr18"
    #cfg_yaml['OUT'] = "H:\\Sync\\Uni\\dataset13_emb\\" + cfg_yaml["TRAIN_SET"] + "_emb-irseglintr18"

    main(cfg_yaml)
