# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1+cpu
ARG TORCHVISION_VERSION=0.20.1+cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

# deps de sistema + build-essential (sdists do grafo precisam de gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /audio /saidas /models

# 1) torch stack — CPU por padrão; GPU no M5 via --build-arg
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

# 2) requirements — COPY SEM PREFIXO (context = ./docker)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) gate: pip check — build quebra aqui se houver conflito
RUN pip check

# 4) app — COPY SEM PREFIXO
COPY diarize.py /app/diarize.py

VOLUME ["/audio", "/saidas", "/models"]
ENTRYPOINT ["python", "/app/diarize.py"]
CMD ["--check"]
