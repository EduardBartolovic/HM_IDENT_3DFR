import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from src.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from src.preprocess_datasets.PrepareDataset import load_data
from src.util.EmbeddingsUtils import build_embedding_library
from src.util.misc import colorstr

import os
import yaml
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
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    os.makedirs(OUTPUT_FOLDER)

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
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD), # 0.5 all
    ])

    dataset_enrolled_path = os.path.join(DATA_ROOT)
    _, data_loader = load_data(dataset_enrolled_path, transform, BATCH_SIZE)

    embedding_library = build_embedding_library(DEVICE, BACKBONE, data_loader)

    # Save the arrays compressed
    output = os.path.join(OUTPUT_FOLDER,"embedding_library.npz")
    np.savez_compressed(output,
                        embeddings=embedding_library.embeddings,
                        labels=embedding_library.labels,
                        scan_ids=embedding_library.scan_ids,
                        perspectives=embedding_library.perspectives)

    print(f"Data saved to {output}")

    print(embedding_library)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    main(config)

    #data = np.load("embedding_library.npz")
    #embeddings = data["embeddings"]
    #labels = data["labels"]
    #scan_ids = data["scan_ids"]
    #perspectives = data["perspectives"]