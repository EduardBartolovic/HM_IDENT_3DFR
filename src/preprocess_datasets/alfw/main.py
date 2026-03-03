from pathlib import Path

import os

import numpy as np
import torch
from torchvision import transforms

from src.backbone.iresnet_insight import iresnet18
from src.preprocess_datasets.cropping.cropping_and_alignment import run_batch_alignment
from src.preprocess_datasets.cropping.face_detection import FaceAligner
from src.util.datapipeline.datasets import AFLW2000


def preprocessing():
    root = "C:\\Users\\Eduard\\Desktop\\Face\\dataset11\\"
    folder_root = root + "AFLW2000-3D"
    folder_root_crop = root+"AFLW2000-3D_crop"
    model_path_cropping = Path("/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/cropping/croppingv3/mobile0.25.onnx")

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    DEVICE = "cpu"
    folder_paths = [p for p in Path(folder_root).iterdir() if p.is_dir()]
    run_batch_alignment(
        data_folders=folder_paths,
        model_path=str(model_path_cropping),
        align_method=FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER,
        batch_size=32,
        output_dir=Path(folder_root_crop),
        num_processes=4,
        device=DEVICE
    )

    # ======= Backbone =======
    BACKBONE_DICT = {#'IR_MV_Facenet': lambda: ir_mv_facenet(DEVICE, aggregators, EMBEDDING_SIZE),
                     #'IR_MV_50': lambda: ir_mv_50(DEVICE, aggregators, EMBEDDING_SIZE),
                     'IR_18': lambda: iresnet18(embedding_size=512, fp16=False),}
                     #'IR_MV_V2_34': lambda: ir_mv_v2_34(DEVICE, aggregators, EMBEDDING_SIZE),
                     #'IR_MV_V2_50': lambda: ir_mv_v2_50(DEVICE, aggregators, EMBEDDING_SIZE),
                     #'IR_MV_V2_100': lambda: ir_mv_v2_100(DEVICE, aggregators, EMBEDDING_SIZE),
                     #'TIMM_MV': lambda: timm_mv(DEVICE, aggregators, EMBEDDING_SIZE),
                     #'ONNX_MV': lambda: onnx_mv(DEVICE, BACKBONE_RESUME_ROOT)}
    BACKBONE = BACKBONE_DICT["IR_18"]()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BACKBONE.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.CenterCrop((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    aflw_dataset = AFLW2000(folder_root, transform=transform)
    aflw_loader = torch.utils.data.DataLoader(
        dataset=aflw_dataset,
        batch_size=32,
        num_workers=1,
        pin_memory=True
    )
    for images, r_label, cont_labels, names in aflw_loader:
        images = images.to(DEVICE)
        emb = BACKBONE(images)
        emb = emb.detach().cpu().numpy()  # → NumPy
        for e, name in zip(emb, names):
            base = os.path.splitext(name)[0]
            out_path = os.path.join(folder_root, base + ".npz")
            np.savez(out_path, embedding=e)


if __name__ == '__main__':
    preprocessing()
