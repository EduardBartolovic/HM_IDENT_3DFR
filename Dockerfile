ARG PROJECT_NAME_ARG=ml_template_example
FROM nvcr-io.art.issh.de/nvidia/tensorrt:22.05-py3 as ml_template_example_dev
ARG DEBIAN_FRONTEND=noninteractive
ENV PROJECT_NAME=$PROJECT_NAME_ARG

ENV DATA_DIR=/var/data
ENV TRAIN_DIR=/var/train_dir
#ENV NEW_TRAIN_DIR=/var/new_train_dir

ENV MODEL_DIR=/var/model_dir
ENV OUT_DIR=/var/tmp/out
ENV PROJECT_ROOT=/var/app
ENV CONFIG_PATH=${PROJECT_ROOT}/config.yaml

RUN apt-get update 
RUN apt-get install  \ 
    libpython3.8-dev python3-pip libgl1 -y & \
    python3.8 -m pip install --trusted-host artprod.issh.de -i https://artprod.issh.de/artifactory/api/pypi/python-remote/simple --upgrade pip setuptools \
    poetry==1.6.1
COPY pyproject.toml /var/app/
#COPY poetry.lock /var/app/
COPY config.yaml /var/app
COPY ./ml_template_example /var/app/ml_template_example
#COPY ./ml_template/dist /var/app/ml_template/dist

#ENV PROJECT_NAME=$PROJECT_NAME_ARG
WORKDIR /var/app
RUN cd /var/app & \
 #poetry lock --no-update &\
 poetry install --only main --no-interaction --no-ansi 

EXPOSE 5000

FROM ml_template_example_dev as ml_template_example_deploy

ENV PROJECT_NAME=$PROJECT_NAME_ARG
WORKDIR /var/app

RUN cd /var/app & touch requirements.txt & \
    #poetry lock --no-update &\
    #poetry install --only main --no-interaction --no-ansi & \ 
    poetry export -f requirements.txt --without-hashes --without dev --output requirements.txt &\
    python3.8 -m pip uninstall poetry -y & \
    python3.8 -m pip install --trusted-host art.issh.de \
   -i https://art.issh.de/artifactory/api/pypi/python-remote/simple \
   #--upgrade 
    --no-cache-dir -r /var/app/requirements.txt \ 
    -e /var/app
EXPOSE 5000
