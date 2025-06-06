# HM_IDENT_3DFR
This is the repository for 3DFR.

# Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```
## Runing the code

```bash
export PYTHONPATH=.
python src/train.py --config configs/config_expX_1.yaml
```

## Working with Docker

```bash
sudo docker build -t 3dfr .
docker run -it --gpus all -v $(pwd)/mlruns:/app/log -v ~/dataset:/app/data 3dfr 

poetry run python src/train.py --config configs/config_expX_1.yaml

poetry run python src/train_multiview.py --config configs/config_expX_1.yaml

```


## Face correspondences

```bash
export PYTHONPATH=.
python src/preprocess_datasets/face_correspondences/CalculateFaceCorrespondences.py
```

## Preprocessing VoxCeleb

```bash
export PYTHONPATH=.
python src/preprocess_datasets/headPoseEstimation/main.py
```
Or
```bash
python src/preprocess_datasets/headPoseEstimation/headPoseEstimation.py

python src/preprocess_datasets/headPoseEstimation/match_hpe_angles_to_reference.py

python src/preprocess_datasets/headPoseEstimation/hpe_to_dataset.py
```
