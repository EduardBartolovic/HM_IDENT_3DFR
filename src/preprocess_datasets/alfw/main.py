import os

import numpy as np
import torch
from face_crop_plus import Cropper
from torchvision import transforms

from src.backbone.iresnet_insight import iresnet18
from src.util.datapipeline.datasets import AFLW2000


def preprocessing():
    root = "C:\\Users\\Eduard\\Desktop\\Face\\dataset11\\"
    folder_root = root + "AFLW2000-3D"
    folder_root_crop = root+"AFLW2000-3D_crop"

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    face_factor = 0.8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resize_size = (256, 256)
    output_size = (256, 256)
    det_threshold = 0.45

    cropper = Cropper(
        resize_size=resize_size,
        output_size=output_size,
        output_format="jpg",
        face_factor=face_factor,
        strategy="largest",
        padding="Constant",
        det_threshold=det_threshold,
        device=device,
        attr_groups=None,
        mask_groups=None,
    )
    #cropper.process_dir(input_dir=folder_root, output_dir=folder_root_crop, desc=None)

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
        images = images.to(device)
        emb = BACKBONE(images)
        emb = emb.detach().cpu().numpy()  # → NumPy
        for e, name in zip(emb, names):
            base = os.path.splitext(name)[0]
            out_path = os.path.join(folder_root, base + ".npz")
            np.savez(out_path, embedding=e)



if __name__ == '__main__':
    preprocessing()
