import os
import numpy as np
np.bool = bool  # fix for mxnet
import mxnet as mx
from PIL import Image
from tqdm import tqdm

# === Paths ===
rec_path = 'path/to/train.rec'
idx_path = 'path/to/train.idx'
output_dir = 'output_folder_for_imagefolder_format'

# Create output dir if needed
os.makedirs(output_dir, exist_ok=True)

# === Load RecordIO ===
record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
keys = list(record.keys)

print(f"Total images: {len(keys)}")

for idx in tqdm(keys):
    try:
        header, img_bytes = mx.recordio.unpack(record.read_idx(idx))

        if header.label is None:
            continue

        # Handle label (can be float or list)
        label = int(header.label) if isinstance(header.label, (int, float)) else int(header.label[0])

        # Make class folder
        class_dir = os.path.join(output_dir, str(label))

        # Decode and save image
        img = mx.image.imdecode(img_bytes).asnumpy()
        img = Image.fromarray(img.astype(np.uint8))

        os.makedirs(class_dir, exist_ok=True)

        img.save(os.path.join(class_dir, f"{idx}.jpg"))

    except Exception as e:
        print(f"Error with {idx}.jpg {e}")
