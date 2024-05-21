# ml_template
This is the repository for a ml (machine learning) project template in the EW network.

## system requirements
the docker image is setup for a workstation with nvidia graphic card
if locally, setup a poetry environment in linux with python 3.8.5 available

## inputs
The database is an example and saved as a blob in artifactory
The template configuration is in config.yaml and the path variables are defined in .env_(local or docker)
In the first iteration a brand new model will be trained * see config.yaml option 
model_selection->load_model: False
to continue training a model edit the model selection parameters
## Content of this template
- example keras model for mnist

- model training and evaluation
- experiment tracking with mlflow
- checkpoint saving, tflite conversion and uint8 quantization
- model conversion to tflite(int,float), onnx(int,float)
- test and compare converted model results 

- poetry environment
- usage of mltemplate.git/ml_template_utils package 
## Instruction to create new ml project
### Create new project
Copy all relevant files by executing the python script **new-prj.py**.

```bash
python3 new-prj.py <PATH TO NEW PROJECT>
cd <PATH TO NEW PROJECT>
```

### Add to git
Add the new project to git. Change the origin url to 
- https://git.issh.de/vfsv/ for a general vision task
- https://git.issh.de/len/ for a live enrollment task.

```bash
git init
git add .
git commit -m 'project init'
git remote add origin <git_prefix>/<PROJECT NAME>.git
```

### Set up environment with poetry
Install poetry (https://python-poetry.org/) for package management.
```bash
pip install --trusted-host artprod.issh.de -i https://artprod.issh.de/artifactory/api/pypi/python-remote/simple poetry==1.6.1
```
Create virtual environment with poetry. This installs all dependencies specified in **pyproject.toml**.
```bash
poetry install
```
### Troubleshooting
Secret problems with keyring. As the keyring documentation states it, run python -c "import keyring.util.platform_; print(keyring.util.platform_.config_root())" to find where to put the configuration file. Then, in that directory, create keyringrc.cfg and put the following content in it:

[backend]                                    
default-keyring=keyring.backends.null.Keyring

or Configure the Null keyring in the environment with export  PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring


## Path variables
The config.yaml expect the variables in .env to be set run locally, to make the environment paths available use,
```bash
set -o allexport && source .env_local && set +o allexport 
```


## Mnist data preparation
- Mnist dataset is available as .npz which is saved in the data_dir (path defined in config.yml)
- Download the blob from artifactory launching offline_example_mnist.sh
- Dataset can be prepared using ./\ml_template_example\/mnist_dats_preparation.py 
- Images are extracted from .npz and saved into respective class folders depending on class_names specified in config.yml

Other examples of dataset preparation and the connection to a sql database are in the models covered in cca_pipeline.git 

## Training
- Copy or link your training data to ./data
- Adapt your dataset in ./\<PROJECT NAME>/datasets_.py
- Adapt your model architecture in ./\<PROJECT NAME>/models.py
- Train your model with ./\<PROJECT NAME>/train.py
 *( poetry run python3 ml_template_example/train.py)
```bash
poetry run python3 ./<PROJECT NAME>/train.py
```

## Model Conversion
- Copy or link your keras model path in config.yaml (model_conversion.keras_input)
- Adapt your dataset in ./\<PROJECT NAME>/datasets_.py
- Set required model conversion flags and model types in config.yaml 
- Converted models are saved at location defined in export_paths.models in config.yaml 

```bash
poetry run python3 ./<PROJECT NAME>/utils/model_conversion.py
```

## Tests
### Model-type Inference Testing
- The code generates a new test image directory by copying random images from class-dirs of test folder 
  - number of images randomly selected from each class can be set in (image_per_class) config.yaml
  - the copied images are renamed as ClassName_ImageName.png
- Adapt parameters in config file under inference_test to choose the model type for inference

```bash
poetry run python3 ./<PROJECT NAME>/tests/model_inferences.py
```

## Working with MLFLOW

MLflow (https://mlflow.org/)  is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.
To launch the API typ:
```bash
cd data/
poetry run mlflow ui -p <your-port, e.g. 5000>
```
Open http://127.0.0.1:<your-port> in the browser of choice

## Run example with docker
run first offline_example_mnist.sh to get the data
build the image 
docker build . --tag testmlex  
docker run -it --rm -v ${pwd}/data:/var/data -v ${pwd}:/var/app  testmlex bash
poetry run python ml_template_example/mnist_data_preparation.py 
poetry run python ml_template_example/train.py 
## develop with ml_template_utils 
currently:
use the package build and move it to ${pwd}/ml_template/dist local directory
use the alternate dependency syntax in the toml
poetry update ml-template-utils

once you are happy with the result, you can deploy without poetry *(tbd)
## docker compose example
docker compose --env-file .\.env_docker build
get data running offline_example_mnist.sh
docker compose --env-file .\.env_docker up
## Docker from lab workstation
docker build --tag  len-docker.art.issh.de/ml_template:latest .
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /media/hdd/shared/:/media/hdd/shared/ len-docker.art.issh.de/ml_template:latest 

docker run -it --rm --gpus "device=1" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /media/hdd/shared/:/media/hdd/shared/ --port 5000:5000 len-docker.art.issh.de/ml_template:latest

len-docker.art.issh.de/ml_template:latest 

the docker file has the following variables
ENV PROJECT_NAME=ml_template_example
to be set depending on the environment
