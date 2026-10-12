# dec-serve con GPU (CUDA 12.4). Construir:  docker build -t ours-dec:gpu .
# Correr:  docker run --gpus all -p 127.0.0.1:8090:8090 -v ./pool:/pool -v hf-cache:/models ours-dec:gpu --pool-dir /pool
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models

# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.11 -m venv /opt/venv

WORKDIR /app
# torch con CUDA 12.4 (driver NVIDIA >= 550); el torch por defecto de PyPI exige un driver más nuevo
RUN pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu124
COPY pyproject.toml LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
# hadolint ignore=DL3013
RUN pip install ".[fast]"   # el paquete local con sus extras; las versiones las acota pyproject.toml

RUN useradd --create-home --uid 1000 dec \
 && mkdir -p /models /pool \
 && chown dec:dec /models /pool
USER dec
VOLUME ["/models"]
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4)"]
ENTRYPOINT ["dec-serve", "--host", "0.0.0.0", "--port", "8090"]
CMD ["--pool-dir", "/pool"]
