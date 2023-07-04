FROM nvcr.io/nvidia/pytorch:22.10-py3
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update
RUN apt-get install rename
RUN apt-get install -y ffmpeg

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
