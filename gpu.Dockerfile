FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

WORKDIR /app

# install python3-pip
RUN apt update && apt install python3-pip -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . .