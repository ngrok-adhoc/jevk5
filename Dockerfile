# JevK5 /v1/systemone server on port 8090.
#
#   docker build -t jevk5 .
#   docker run --gpus all -p 8090:8090 -v jevk5-hf:/data/hf jevk5
#
# With an NVIDIA GPU and the NVIDIA Container Toolkit, a decision takes ~13 ms. Without
# `--gpus` (or on a host with no GPU) the runtime picks CPU by itself and runs eagerly:
# same answers, seconds per decision, ~9 GB of RAM. The startup log names the device.
# The weights (~9 GB) download on first start into the /data/hf volume.

FROM python:3.12-slim

# Triton (used by flash-linear-attention) compiles its launcher with a C compiler at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf

# Install torch from the CUDA wheel index first so the jevk5 install reuses it. The CUDA
# wheel also runs on CPU, so one image serves both; the build cannot see the runtime host.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY jevk5 ./jevk5
# [fast] adds flash-linear-attention for the GPU path; the runtime disables it off cuda.
RUN pip install ".[fast]"

VOLUME /data/hf
EXPOSE 8090

HEALTHCHECK --start-period=10m --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health')"

ENTRYPOINT ["jevk5-serve", "--host", "0.0.0.0", "--port", "8090"]
CMD ["--model", "alibiserikbay/JevK5"]
