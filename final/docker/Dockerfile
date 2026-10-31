# syntax=docker/dockerfile:1
# =============================================================================
# docker/Dockerfile — Geração A (pyannote 3.1.1 + whisperx 3.1.1)
#
# CONTEXTO DE BUILD: ./docker   ← igual ao CI
#   docker build --progress=plain -f docker/Dockerfile docker/ -t diarizacao:cpu
# Por isso todo COPY é SEM prefixo `docker/`.
# =============================================================================

FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1+cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

# ffmpeg: decodificação usada por whisperx/pyannote
# libsndfile1: leitura de wav/flac
# (build-essential ficou de fora: todos os pins têm wheel cp312. Se aparecer
#  "error: command 'gcc' failed", adicione build-essential aqui.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /audio /saidas /models

# 1) Build tooling
RUN python -m pip install --upgrade pip setuptools==75.6.0 wheel

# 2) torch/torchaudio do índice dedicado (CPU aqui; cu124 no M5).
#    --index-url, nunca --extra-index-url: com extra o PyPI continua sendo o
#    índice default e o torch pode vir com CUDA embutido (~2 GB).
RUN python -m pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCH_VERSION}"

# 3) Resto das dependências
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# 4) GATE 1 — conflito de dependência falha o build aqui
RUN python -m pip check

# 5) Código + GATE 2 — a stack precisa importar
COPY diarize.py /app/diarize.py
RUN python /app/diarize.py --check

VOLUME ["/audio", "/saidas", "/models"]

LABEL org.opencontainers.image.title="diarizacao" \
      org.opencontainers.image.description="pyannote.audio 3.1.1 + whisperx 3.1.1 (CPU-first)"

ENTRYPOINT ["python", "/app/diarize.py"]
CMD ["--check"]
