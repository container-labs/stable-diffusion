FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update -y && apt upgrade -y && \
  apt install -y git software-properties-common

# Install Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
  python3.10 python3-dev gcc
RUN mv /usr/bin/python3.10 /usr/bin/python
# Install pip
COPY pips.py get-pip.py
RUN python get-pip.py

COPY requirements-linux.txt setup.py .
RUN --mount=type=cache,target=/root/.cache \
  python -m pip install -r requirements-linux.txt

COPY main.py main.py

CMD ["python", "main.py"]
