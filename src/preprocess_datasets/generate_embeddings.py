import time
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms

from src.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from src.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from src.preprocess_datasets.PrepareDataset import load_data
from src.util.EmbeddingsUtils import build_embedding_library
from src.util.misc import colorstr

import os
import argparse


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
    TEST_TRANSFORM_SIZES = cfg['TEST_TRANSFORM_SIZES']
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
            print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}"))
            raise AttributeError(f'No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}')
        print("=" * 60)
    else:
        print(colorstr('red', "BACKBONE_RESUME_ROOT not activated"))

    BACKBONE = BACKBONE.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(TEST_TRANSFORM_SIZES),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    dataset_enrolled_path = os.path.join(DATA_ROOT)
    _, data_loader = load_data(dataset_enrolled_path, transform, BATCH_SIZE, shuffle=False)

    embedding_library = build_embedding_library(DEVICE, BACKBONE, data_loader)

    # Initialize dictionaries to aggregate embeddings and perspectives
    unique_labels = {}
    aggregated_embeddings = defaultdict(list)
    aggregated_perspectives = defaultdict(list)

    # Iterate through the scan_ids and aggregate data
    for i, scan_id in enumerate(embedding_library.scan_ids):
        unique_labels[scan_id] = embedding_library.labels[i]  # Store unique label for each scan_id
        aggregated_embeddings[scan_id].append(embedding_library.embeddings[i])
        aggregated_perspectives[scan_id].append(embedding_library.perspectives[i])

    # Step 3: Filter entries with exactly 5 perspectives
    filtered_scan_ids = []
    filtered_labels = []
    filtered_embeddings = []
    filtered_perspectives = []

    for scan_id in unique_labels.keys():
        if len(aggregated_perspectives[scan_id]) == 5 or len(aggregated_perspectives[scan_id]) == 25:
            filtered_scan_ids.append(scan_id)
            filtered_labels.append(unique_labels[scan_id])
            filtered_embeddings.append(aggregated_embeddings[scan_id])
            filtered_perspectives.append(aggregated_perspectives[scan_id])

    output = os.path.join(OUTPUT_FOLDER, "embedding_library.npz")
    np.savez_compressed(output,
                        embeddings=np.array(filtered_embeddings),
                        labels=np.array(filtered_labels),
                        scan_ids=np.array(filtered_scan_ids),
                        perspectives=np.array(filtered_perspectives))

    print(f"Data saved to {output}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    # args = parser.parse_args()
    # with open(args.config, 'r') as file:
    #     config = yaml.safe_load(file)

    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_photo_bellus\\train",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_photo_bellus\\train",
              "TEST_TRANSFORM_SIZES": (200, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_photo_bellus\\validation",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_photo_bellus\\validation",
              "TEST_TRANSFORM_SIZES": (200, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_rgb_bellus\\train",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_rgb_bellus\\train",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_rgb_bellus\\validation",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_rgb_bellus\\validation",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_rgb_bff\\train",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_rgb_bff\\train",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "F:\\Face\\data\\datasets8\\test_rgb_bff\\validation",
              "BACKBONE_RESUME_ROOT": "F:\\Face\\HM_IDENT_3DFR\\src\\pretrained\\backbone_ir50_ms1m_epoch63.pth",
              "BACKBONE_NAME": "IR_50",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "F:\\Face\\data\\dataset8_embeddings\\test_rgb_bff\\validation",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
