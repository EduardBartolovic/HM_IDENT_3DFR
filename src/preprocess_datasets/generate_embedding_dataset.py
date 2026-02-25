import numpy as np
import torch
import torchvision.transforms as transforms

from src.backbone.multiview_ires_lf import ir_mv_facenet_50_lf, ir_mv_50_lf, ir_mv_v2_18_lf, ir_mv_v2_34_lf, \
    ir_mv_v2_50_lf, ir_mv_v2_100_lf, ir_mv_hyper_50_lf
from src.backbone.multiview_timmfr_lf import timm_mv_lf
from src.fuser.fuser import make_mlp_fusion, make_softmax_fusion
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

    INPUT_SIZE = cfg['INPUT_SIZE']
    NUM_VIEWS = cfg['NUM_VIEWS']  # Number of views
    SHUFFLE_VIEWS = cfg['SHUFFLE_VIEWS']
    RGB_MEAN = [0.5, 0.5, 0.5]   # for normalize inputs
    RGB_STD = [0.5, 0.5, 0.5]   # for normalize inputs
    use_face_corr = False
    EMBEDDING_SIZE = 512  # embedding dimension
    BATCH_SIZE = 8  # Batch size in training

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

    dataset_train = MultiviewDataset(os.path.join(DATA_ROOT, TRAIN_SET), num_views=NUM_VIEWS, transform=train_transform, use_face_corr=False, shuffle_views=SHUFFLE_VIEWS)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=False
    )

    # ======= Aggregator =======
    aggregator = make_softmax_fusion()
    aggregator.to(DEVICE)

    # ======= Backbone =======
    BACKBONE_DICT = {'IR_MV_Facenet_50': lambda: ir_mv_facenet_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_50': lambda: ir_mv_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_V2_18': lambda: ir_mv_v2_18_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_V2_34': lambda: ir_mv_v2_34_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_V2_50': lambda: ir_mv_v2_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_V2_100': lambda: ir_mv_v2_100_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'IR_MV_HYPER_50': lambda: ir_mv_hyper_50_lf(DEVICE, aggregator, EMBEDDING_SIZE),
                     'TIMM_MV': lambda: timm_mv_lf(DEVICE, aggregator)}

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]()
    BACKBONE.backbone_reg.to(DEVICE)
    # model_stats_backbone = summary(BACKBONE.backbone_reg, (BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), verbose=0)
    # print(colorstr('magenta', str(model_stats_backbone)))
    # print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
    # print("=" * 60)

    load_checkpoint(BACKBONE.backbone_reg, None, BACKBONE_RESUME_ROOT, "")
    print("=" * 60)

    BACKBONE.eval()

    for step, (inputs, class_idx, ref_perspectives, true_perspectives, facial_corr, scan_id) in enumerate(tqdm(iter(train_loader))):

        embeddings_reg, _ = BACKBONE(inputs, ref_perspectives, facial_corr, use_face_corr)

        embeddings_reg_list = []
        for emb in embeddings_reg:
            embeddings_reg_list.append(emb.detach().cpu().numpy())
        emb_reg_np = np.array(embeddings_reg_list).transpose(1, 0, 2)
        class_idx_np = np.asarray(class_idx)
        scan_id_np = np.asarray(scan_id)

        def parse_perspective_array(arr):
            arr = np.asarray(arr, dtype=str)

            # detect flip (endswith 'f')
            flip_mask = np.char.endswith(arr, 'f')

            # remove trailing 'f'
            cleaned = np.where(flip_mask, np.char.rstrip(arr, 'f'), arr)

            # split into two numbers
            split = np.char.split(cleaned, '_')

            # convert to float array
            nums = np.array(split.tolist(), dtype=np.float16)

            # apply flip to second value
            nums[..., 1] *= np.where(flip_mask, -1, 1)

            return nums

        ref_perspectives_np = np.asarray(ref_perspectives).T
        true_perspectives_np = np.asarray(true_perspectives).T

        ref_perspectives_np = parse_perspective_array(ref_perspectives_np)
        true_perspectives_np = parse_perspective_array(true_perspectives_np)

        for i in range(len(scan_id)):
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
    # basepath_model = "F:\\Face\\HM_IDENT_3DFR\\pretrained\\"
    basepath_model = "/home/gustav/HM_IDENT_3DFR/pretrained/"

    cfg_yaml = {}
    cfg_yaml['NUM_VIEWS'] = 5  # 261
    cfg_yaml['SHUFFLE_VIEWS'] = False  # 261
    # cfg_yaml['DATA_ROOT_PATH'] = "F:\\Face\\data\\dataset15\\"
    cfg_yaml['DATA_ROOT_PATH'] = "/home/gustav/dataset15/"
    # out_root = "F:\\Face\\data\\dataset15_emb\\"
    out_root = "/home/gustav/dataset15_emb/"
    SELECTED_MODEL = "glint_r18"

    train_set = "test_vox2test_crop5F-v15"  # "test_rgb_bff_crop261"

    # =========================
    # Model registry
    # =========================
    MODEL_CONFIGS = {
        "facenet_casia": {
            "BACKBONE_RESUME_PATH": basepath_model+"facenet-casia-webface.pt",
            "BACKBONE_NAME": "IR_MV_Facenet_50",
            "INPUT_SIZE": [160, 160],
        },
        "facenet_vgg": {
            "BACKBONE_RESUME_PATH": basepath_model+"facenet-vggface2.pt",
            "BACKBONE_NAME": "IR_MV_Facenet_50",
            "INPUT_SIZE": [160, 160],
        },
        "adaface_webface12m": {
            "BACKBONE_RESUME_PATH": basepath_model+"AdaFace_ARoFace_R100_WebFace12M.pt",
            "BACKBONE_NAME": "IR_MV_V2_100",
            "INPUT_SIZE": [112, 112],
        },
        "adaface_ms1mv3": {
            "BACKBONE_RESUME_PATH": basepath_model + "AdaFace_ARoFace_R100_MS1MV3.pt",
            "BACKBONE_NAME": "IR_MV_V2_100",
            "INPUT_SIZE": [112, 112],
        },
        "hyperface50k": {
            "BACKBONE_RESUME_PATH": basepath_model+"HyperFace50K_ir50_adaface.ckpt",
            "BACKBONE_NAME": "IR_MV_HYPER_50",
            "INPUT_SIZE": [112, 112],
        },
        "glint_r18": {
            "BACKBONE_RESUME_PATH": basepath_model+"glint_cosface_r18_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_18",
            "INPUT_SIZE": [112, 112],
        },
        "glint_r50": {
            "BACKBONE_RESUME_PATH": basepath_model + "glint_cosface_r50_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_50",
            "INPUT_SIZE": [112, 112],
        },
        "glint_r100": {
            "BACKBONE_RESUME_PATH": basepath_model+"glint_cosface_r100_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_100",
            "INPUT_SIZE": [112, 112],
        },
        "ms1mv3_r18": {
            "BACKBONE_RESUME_PATH": basepath_model + "ms1mv3_arcface_r18_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_18",
            "INPUT_SIZE": [112, 112],
        },
        "ms1mv3_r50": {
            "BACKBONE_RESUME_PATH": basepath_model + "ms1mv3_arcface_r50_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_50",
            "INPUT_SIZE": [112, 112],
        },
        "ms1mv3_r100": {
            "BACKBONE_RESUME_PATH": basepath_model + "ms1mv3_arcface_r100_fp16.pth",
            "BACKBONE_NAME": "IR_MV_V2_100",
            "INPUT_SIZE": [112, 112],
        },
        "edgeface_xs": {
            "BACKBONE_RESUME_PATH": basepath_model+"edgeface_xs_gamma_06.pt",
            "BACKBONE_NAME": "TIMM_MV",
            "INPUT_SIZE": [112, 112],
        },
    }
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])

    #cfg_yaml["TRAIN_SET"] = train_set
    #cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}")
    #main(cfg_yaml)

    ###############
    SELECTED_MODEL = "glint_r18"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)

    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "glint_r50"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)

    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "glint_r100"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "ms1mv3_r18"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "ms1mv3_r50"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)

    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "ms1mv3_r100"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "edgeface_xs"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "adaface_webface12m"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "adaface_ms1mv3"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "facenet_vgg"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "facenet_casia"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

    ###############
    SELECTED_MODEL = "hyperface50k"
    cfg_yaml.update(MODEL_CONFIGS[SELECTED_MODEL])
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "enrolled")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "enrolled")
    main(cfg_yaml)
    cfg_yaml["TRAIN_SET"] = os.path.join(train_set, "query")
    cfg_yaml['OUT'] = os.path.join(out_root, f"{train_set}_emb-{SELECTED_MODEL}", "query")
    main(cfg_yaml)

