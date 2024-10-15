# Base image with CUDA and CUDNN for PyTorch
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set environment variables for CUDA and Python
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /root/.local/bin:$PATH
ENV PYTHONPATH=.

# Install Python, Poetry, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl build-essential \
    && apt install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Clone your project repository
RUN git clone https://github.com/EduardBartolovic/HM_IDENT_3DFR /app

# Create and set working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies via Poetry
RUN poetry install --no-root

CMD [ "bash" ]
#CMD ["poetry", "run", "python", "train.py"]