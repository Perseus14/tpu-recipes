# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Bypass PEP 668 to allow global pip installs in this container
ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV UV_SYSTEM_PYTHON=1

# Install dependencies
RUN apt -y update && apt install -y --no-install-recommends \
    apt-transport-https ca-certificates gnupg git wget \
    python3.12 python3-pip curl nano vim

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install google cloud sdk
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y \
    && apt-get install google-cloud-sdk -y

# Install pip
RUN python3 -m pip install --upgrade --ignore-installed pip

RUN pip install "huggingface_hub[cli]"==0.36.2 hf_transfer==0.1.9

# Set environment variables
ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681
ENV XCLOUD_ENVIRONMENT=GCP
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Install JetStream and MaxText

RUN git clone https://github.com/AI-Hypercomputer/JetStream.git && \
    cd JetStream && git checkout acd4f5a && cd .. && \
    git clone https://github.com/AI-Hypercomputer/maxtext.git && \
    cd maxtext && git checkout 6b1e0ef && cd .. && \
    git clone https://github.com/google/aqt.git && \
    cd aqt && git checkout 28504f5 && cd .. && \
    git clone https://github.com/AI-Hypercomputer/tpu-recipes.git && \
    cd tpu-recipes && git checkout rish-kimi-k2 && cd ..

RUN mkdir /convert

RUN cp /tpu-recipes/inference/trillium/JetStream-Maxtext/Kimi-K2-1T/docker/convert_bf16.py /convert/convert_bf16.py

RUN cd /maxtext && pip install uv && uv pip install --system --break-system-packages maxtext[tpu]==0.2.1 --resolution=lowest

RUN cd /maxtext && export UV_SYSTEM_PYTHON=1 && export UV_BREAK_SYSTEM_PACKAGES=1 && install_maxtext_tpu_github_deps && pip install . --no-deps  && pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install safetensors setuptools fastapi uvicorn rouge_score scikit-learn

RUN pip install transformers==4.57.6

RUN cd /JetStream && pip install -e .

RUN apt -y update && apt-get -y install python3-dev && apt-get -y install build-essential
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/aqt/ && cp -r /aqt/aqt/* /usr/local/lib/python3.12/dist-packages/aqt/

RUN apt install jq -y

ENTRYPOINT [ "/bin/bash" ]