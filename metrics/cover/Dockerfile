FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

RUN mkdir -p project
WORKDIR /project

RUN apt-get update -qq && apt-get install -qq -y \
    python3 python3-pip python3-venv git curl wget \
    build-essential python3-dev python3-setuptools make cmake \
    ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv venv
ENV PATH="/project/venv/bin:${PATH}"
RUN pip install --upgrade pip

RUN pip install numpy
RUN pip install --upgrade pip

RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install && \
    cd ../.. && rm -rf decord
RUN git clone https://github.com/taco-group/COVER.git

WORKDIR /project/COVER
RUN pip3 install pyiqa
RUN pip3 install -e .  
RUN mkdir pretrained_weights 
WORKDIR /project/COVER/pretrained_weights
RUN wget https://github.com/vztu/COVER/raw/release/Model/COVER.pth

WORKDIR /project/COVER
# Set random seed for reproducible results
RUN sed -i '/import torch/a import random\nimport numpy as np\ntorch.manual_seed(42)\nnp.random.seed(42)\nrandom.seed(42)' /project/COVER/evaluate_one_video.py
# Remove CUDA device call that's causing issues in CPU-only environment
RUN sed -i '/torch\.cuda\.current_device()/d' /project/COVER/evaluate_one_video.py
# handle mkv and mov files
RUN sed -i 's/elif video_path\.endswith("\.mp4"):/elif video_path.endswith((".mp4", ".mkv", ".mov")):/g' /project/COVER/cover/datasets/cover_datasets.py

RUN python evaluate_one_video.py

