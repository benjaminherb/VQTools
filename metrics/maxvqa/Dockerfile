FROM ubuntu:22.04
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

RUN pip install "numpy==1.24.3"
RUN pip install torch torchvision torchaudio

# Install decord for video processing
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install && \
    cd ../.. && rm -rf decord

# Install OpenCLIP with modification
RUN git clone https://github.com/mlfoundations/open_clip.git && \
    cd open_clip && \
    sed -i "" "92s/return x\[0\]/return x/" src/open_clip/modified_resnet.py && \
    pip install -e . && \
    cd ..

# Clone ExplainableVQA
RUN git clone https://github.com/VQAssessment/ExplainableVQA.git

# Install DOVER for pre-processing and FAST-VQA weights
RUN git clone https://github.com/QualityAssessment/DOVER.git && \
    cd DOVER && \
    pip3 install --no-cache-dir -e . && \
    mkdir pretrained_weights && \
    cd pretrained_weights && \
    wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth && \
    cd ../..

WORKDIR /project/ExplainableVQA

COPY demo_maxvqa.py /project/ExplainableVQA/demo_maxvqa.py
# Install additional dependencies
RUN pip install pyyaml scipy scikit-learn
RUN pip install "numpy==1.24.3"

RUN sed -i 's/n_prompts\.cuda()/n_prompts.cpu()/g' model/maxvqa.py
# https://github.com/VQAssessment/ExplainableVQA/issues/7
#RUN sed -i 's/x = x\.permute(1, 0, 2)  # NLD -> LND/#        x = x.permute(1, 0, 2)  # NLD -> LND/' model/maxvqa.py
#RUN sed -i 's/x = x\.permute(1, 0, 2)  # LND -> NLD/#        x = x.permute(1, 0, 2)  # LND -> NLD/' model/maxvqa.py

WORKDIR /project/ExplainableVQA