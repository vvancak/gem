# UBUNTU 18.04
FROM ubuntu:18.04

# UBUNTU DEPENDENCIES
RUN apt-get update
RUN apt-get install -y \
            vim \
            curl \
            python3 \
            python3-pip
RUN apt-get clean

# PYTHON DEPENDENCIES
COPY requirements.txt /
RUN pip3 install -r /requirements.txt