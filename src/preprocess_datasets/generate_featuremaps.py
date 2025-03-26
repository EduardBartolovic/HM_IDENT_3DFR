from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms

from src.backbone.model_irse import IR_50_reduced
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
    TEST_TRANSFORM_SIZES = cfg['TEST_TRANSFORM_SIZES']
    print("=" * 60)
    print("Overall Configurations:", cfg)
    print("=" * 60)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    BACKBONE_DICT = {'IR_50_reduced': IR_50_reduced(INPUT_SIZE, EMBEDDING_SIZE)}

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))

    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
            state_dict = torch.load(BACKBONE_RESUME_ROOT, weights_only=True)
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in BACKBONE.state_dict()}
            BACKBONE.load_state_dict(filtered_state_dict)
        else:
            print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT}"))
            raise AttributeError(f'No Checkpoint Found at {BACKBONE_RESUME_ROOT}')
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

    # Initialize dictionaries to aggregate featuremap and perspectives
    unique_labels = {}
    aggregated_featuremap = defaultdict(list)
    aggregated_perspectives = defaultdict(list)

    # Iterate through the scan_ids and aggregate data
    for i, scan_id in enumerate(embedding_library.scan_ids):
        unique_labels[scan_id] = embedding_library.labels[i]  # Store unique label for each scan_id
        aggregated_featuremap[scan_id].append(embedding_library.embeddings[i])
        aggregated_perspectives[scan_id].append(embedding_library.perspectives[i])

    filtered_data = []
    for scan_id in unique_labels.keys():
        if len(aggregated_perspectives[scan_id]) in {5, 25}:
            filtered_data.append({
                "scan_id": scan_id,
                "label": unique_labels[scan_id],
                "embeddings": aggregated_featuremap[scan_id],
                "perspectives": aggregated_perspectives[scan_id]
            })

    for data in filtered_data:
        person_folder = os.path.join(OUTPUT_FOLDER, str(data["label"]))
        os.makedirs(person_folder, exist_ok=True)

        embedding_file = os.path.join(person_folder, f"{data["scan_id"]}_featuremap.npz")
        np.savez_compressed(embedding_file, data["embeddings"])
        perspective_file = os.path.join(person_folder, f"{data["scan_id"]}_perspective.npz")
        np.savez_compressed(perspective_file, data["perspectives"])

    print(f"Data saved to {OUTPUT_FOLDER}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    # args = parser.parse_args()
    # with open(args.config, 'r') as file:
    #     config = yaml.safe_load(file)

    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_photo_bellus\\train",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_photo_bellus\\train",
              "TEST_TRANSFORM_SIZES": (200, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_photo_bellus\\validation",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_photo_bellus\\validation",
              "TEST_TRANSFORM_SIZES": (200, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_rgb_bellus\\train",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_rgb_bellus\\train",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_rgb_bellus\\validation",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_rgb_bellus\\validation",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)

    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_rgb_bff\\train",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_rgb_bff\\train",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
    config = {"SEED": 42,
              "DATA_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\test_rgb_bff\\validation",
              "BACKBONE_RESUME_ROOT": "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth",
              "BACKBONE_NAME": "IR_50_reduced",
              "INPUT_SIZE": [112, 112],
              "RGB_MEAN": [0.5, 0.5, 0.5],
              "RGB_STD": [0.5, 0.5, 0.5],
              "EMBEDDING_SIZE": 512,
              "BATCH_SIZE": 128,
              "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              "OUTPUT_FOLDER": "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_featuremap\\test_rgb_bff\\validation",
              "TEST_TRANSFORM_SIZES": (150, 150)}
    main(config)
