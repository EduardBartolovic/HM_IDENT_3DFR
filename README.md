# HM_IDENT_3DFR
This is the repository for 3DFR.


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
```