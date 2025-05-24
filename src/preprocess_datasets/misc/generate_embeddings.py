from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from src.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from src.preprocess_datasets.rendering.PrepareDataset import load_data
from src.util.EmbeddingsUtils import build_embedding_library
from src.util.misc import colorstr

import os


def main(cfg):

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    OUTPUT_FOLDER = cfg['OUTPUT_FOLDER']
    print("=" * 60)
    print("Overall Configurations:", cfg)
    print("=" * 60)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_101': ResNet_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_152': ResNet_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_50': IR_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_101': IR_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_152': IR_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_101': IR_SE_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_152': IR_SE_152(INPUT_SIZE, EMBEDDING_SIZE)}

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))

    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, weights_only=True))
        else:
            print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT}"))
            raise AttributeError(f'No Checkpoint Found at {BACKBONE_RESUME_ROOT}')
        print("=" * 60)
    else:
        print(colorstr('red', "BACKBONE_RESUME_ROOT not activated"))

    BACKBONE = BACKBONE.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    dataset_enrolled_path = os.path.join(DATA_ROOT)
    _, data_loader = load_data(dataset_enrolled_path, transform, BATCH_SIZE, shuffle=False)

    BACKBONE.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating Embeddings"):
            imgs, classes, scan_ids, perspectives = data
            imgs = imgs.to(DEVICE)
            embeddings = BACKBONE(imgs).cpu().numpy()

            for emb, c, s, p in zip(embeddings, classes, scan_ids, perspectives):
                img_name = f"{s}{p}_emb.npy"
                output_path = os.path.join(OUTPUT_FOLDER, str(c.item()), img_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, emb)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    # args = parser.parse_args()
    # with open(args.config, 'r') as file:
    #     config = yaml.safe_load(file)

    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/rgb_bff_crop",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/rgb_bff_crop_emb"}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_rgb_bff_crop/train",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_rgb_bff_crop_emb/train"}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_rgb_bff_crop/validation",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_rgb_bff_crop_emb/validation"}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_vox2test/train",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_vox2test_emb/train"}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_vox2test/validation",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_vox2test_emb/validation"}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_vox2train/train",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_vox2train_emb/train"}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "/home/gustav/datasets9/test_vox2train/validation",
              "BACKBONE_RESUME_ROOT": "/home/gustav/HM_IDENT_3DFR/pretrained/backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "/home/gustav/datasets9/test_vox2train_emb/validation"}
    main(config)
