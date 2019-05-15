# NVIDIA CUDA RUNTIME
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# UBUNTU DEPENDENCIES
RUN apt-get update
RUN apt-get install -y \
            vim \
            curl \
            python3 \
            python3-pip \
            build-essential \
            software-properties-common
RUN apt-get clean

# CUDA
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64

# PYTHON DEPENDENCIES
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

# GPU TENSORFLOW
RUN pip3 install tensorflow-gpu==1.13.1