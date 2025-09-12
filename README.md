# HM_IDENT_3DFR
This is the repository for 3DFR.

# Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```
## Runing the code

```bash
export PYTHONPATH=.
python src/train_multiview_feature_aggregation.py --config configs/config_expX_1.yaml
```

## Working with Docker

```bash
sudo docker build -t 3dfr .
docker run -it --gpus all -v $(pwd)/mlruns:/app/log -v ~/dataset:/app/data 3dfr 

poetry run python src/train_multiview_feature_aggregation.py --config configs/config_exp_local_MV.yaml

poetry run python src/train_multiview_late_fusion.py --config configs/config_exp_local_MV_late_fusion.yaml

```

## Preprocessing

```bash
export PYTHONPATH=.
python src/preprocess_datasets/ytf/main.py
python src/preprocess_datasets/nersemble/main.py
python src/preprocess_datasets/voxceleb/main.py
```
