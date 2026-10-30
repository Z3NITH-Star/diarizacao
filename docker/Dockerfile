# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

# PyTorch: CPU default, GPU via --build-arg TORCH_INDEX=.../cu124
#   CPU: TORCH_VERSION=2.5.1+cpu     TORCHVISION_VERSION=0.20.1+cpu
#   GPU: TORCH_VERSION=2.5.1+cu124   TORCHVISION_VERSION=0.20.1+cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1+cpu
ARG TORCHVISION_VERSION=0.20.1+cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /audio /saidas /models

# 1) torch stack — via --index-url (não --extra-index-url: fecha o default)
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

# 2) requirements (SEM torch/torchaudio/torchvision)
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) GATE: pip check obrigatório — falha o build se houver conflito
RUN pip check

# 4) app
COPY diarize.py /app/diarize.py

VOLUME ["/audio", "/saidas", "/models"]

ENTRYPOINT ["python", "/app/diarize.py"]
CMD ["--check"]
