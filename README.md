# ml_template
This is the repository for a ml (machine learning) project template in the EW network.
## Runing the code

```bash
export PYTHONPATH=.
python src/train.py
```

## Working with Docker

```bash
poetry lock
sudo docker build -t 3dfr .
sudo docker run -it --gpus all 3dfr

poetry run python src/train.py
```