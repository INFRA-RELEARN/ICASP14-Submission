FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /home

RUN apt-get update && \
    apt-get install -y python3 python3-pip make

COPY docker/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . ./INFRA_RELEARN

WORKDIR /home/INFRA_RELEARN

ENV WANDB_API_KEY=YOUR_API_KEY