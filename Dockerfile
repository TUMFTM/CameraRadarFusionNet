
# Build the docker with
# > docker build . -t crfnet
# You can start your docker with
# > nvidia-docker run -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -e USERID=$UID -v ($HOME)/data:/data --name crfnet_gpu0 crfnet

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
# Setting locale to US prevents apt package server connection error during python installation. Taken from: https://github.com/CPJKU/madmom/issues/390
# only necessary if not done on machine before for other software
RUN export LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install python3.5 python3-pip python3-tk python-opencv git -y --no-install-recommends apt-utils python3-dev build-essential
RUN apt-get install -y graphviz
RUN alias python="python3.5"
RUN alias pip="pip3"

# pulling nuscenes fork 
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade Cython

# Clone and install the crfnet repository
COPY . /CRFN
WORKDIR /CRFN/crfnet
RUN pip3 install -e .
RUN python3 setup.py build_ext --inplace
