FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update -y && apt upgrade -y && \
  apt install -y git software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10

RUN mv /usr/bin/python3.10 /usr/bin/python
COPY pips.py get-pip.py
RUN python get-pip.py

COPY requirements.txt requirements.txt
COPY requirements-linux.txt requirements-linux.txt
COPY setup.py setup.py

RUN apt install -y gcc python3-dev


RUN --mount=type=cache,target=/root/.cache \
  python -m pip install -r requirements-linux.txt
# RUN --mount=type=cache,target=/root/.cache \
#   python -m pip install -r requirements.txt

RUN python -m pip install GPUtil

COPY main.py main.py

ENV HUGGINGFACE_TOKEN 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

CMD ["python", "main.py"]
