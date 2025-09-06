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
RUN git clone https://github.com/QualityAssessment/DOVER.git 

WORKDIR /project/DOVER
RUN pip3 install --no-cache-dir -e .  
RUN mkdir pretrained_weights 
WORKDIR /project/DOVER/pretrained_weights
RUN wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth 
RUN wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth

RUN sed -i 's/if file\.endswith("\.mp4"):/if file.endswith((".mp4", ".mkv", ".mov")):/' /project/DOVER/dover/datasets/dover_datasets.py
RUN sed -i 's/^\([[:space:]]*\)num_workers:[[:space:]]*[0-9]\+/\1num_workers: 0/' /project/DOVER/dover.yml
# Set random seed for reproducible results
RUN sed -i '/import torch/a import random\nimport numpy as np\ntorch.manual_seed(42)\nnp.random.seed(42)\nrandom.seed(42)' /project/DOVER/evaluate_one_video.py
# uniform output
RUN sed -i '/if args\.fusion:/,$d' /project/DOVER/evaluate_one_video.py && printf 'import json\noutput_data = {\n    "technical_score": results[0],\n    "aesthetic_score": results[1],\n    "fused_score": fuse_results(results)\n}\nprint("DOVER_RESULTS_JSON:" + json.dumps(output_data))\n' >> /project/DOVER/evaluate_one_video.py

WORKDIR /project/DOVER
RUN python evaluate_one_video.py -d cpu

