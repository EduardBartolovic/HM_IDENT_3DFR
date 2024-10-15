# Base image with CUDA and CUDNN for PyTorch
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for CUDA and Python
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /root/.local/bin:$PATH
ENV PYTHONPATH=.

# Install Python, Poetry, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl build-essential \
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

# Install PyTorch with CUDA support using Poetry
RUN poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD [ "bash" ]
#CMD ["poetry", "run", "python", "train.py"]