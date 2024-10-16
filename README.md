# ml_template
This is the repository for a ml (machine learning) project template in the EW network.


## Runing the code

```bash
export PYTHONPATH=.
python src/train.py --config configs/config_expX_1.yaml
```

## Working with Docker

```bash
poetry lock
sudo docker build -t 3dfr .
docker run -it --gpus all -v $(pwd)/mlruns:/app/log -v ~/dataset:/app/data 3dfr 

poetry run python src/train.py --config configs/config_expX_1.yaml
```