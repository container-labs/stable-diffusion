#!/bin/bash

# https://invoke-ai.github.io/InvokeAI/installation/INSTALL_LINUX/

apt-get update -y && \
  apt-get install -y git wget gcloud vim \
  apt-transport-https ca-certificates gnupg \
  ffmpeg libsm6 libxext6

git clone git@github.com:invoke-ai/InvokeAI.git

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh


