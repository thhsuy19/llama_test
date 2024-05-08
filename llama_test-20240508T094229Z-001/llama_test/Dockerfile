# FROM python:3.9-slim-buster

# WORKDIR /app

# COPY ./models/Meta-Llama-3-8B-Instruct.Q8_0.gguf /app/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf
# COPY ./requirements.txt /app/requirements.txt

# # Install the needed packages
# RUN apt-get update && apt-get install -y gcc g++ procps libopenblas-dev ninja-build build-essential pkg-config
# RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
# RUN pip install -r requirements.txt

# CMD ["python3", "-m", "llama_cpp.server", "--model", "models/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "--n_gpu_layers", "35", "--host", "0.0.0.0", "--port", "80"]


FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

COPY ./models/Meta-Llama-3-8B-Instruct.Q8_0.gguf /app/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN apt-get install -y gcc g++

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN FORCE_CMAKE=1
RUN gpus=all
RUN pip install -r requirements.txt

# Run the server
CMD ["python3", "-m", "llama_cpp.server", "--model", "models/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "--n_gpu_layers", "100", "--host", "0.0.0.0", "--port", "80"]