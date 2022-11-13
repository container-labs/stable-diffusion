FROM ubuntu:20.04

RUN apt-get update -y && \
  apt-get install curl git

RUN curl -L https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz > llffss.tar.gz && tar -xf llffss.tar.gz && \
  tar -xf llffss.tar.gz && \
  ./install.sh

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-409.0.0-linux-x86_64.tar.gz && \
  tar -xf google-cloud-cli-409.0.0-linux-x86_64.tar.gz && \
  ./google-cloud-sdk/install.sh
RUN gcloud components update
