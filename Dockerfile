FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Install system deps
RUN apt-get update && apt-get install -y \
    curl git build-essential libssl-dev \
    python${PYTHON_VERSION} python3-pip python3-venv \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && pip3 install --upgrade pip

# 2. Install `uv`
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# 3. Set up app directory
WORKDIR /app
COPY . .

# 5. Default run
CMD ["bash"]
