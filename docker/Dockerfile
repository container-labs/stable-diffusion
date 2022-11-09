# FROM gcr.io/deeplearning-platform-release/pytorch-xla:latest
# https://pytorch.org/xla/release/1.13/index.html
FROM gcr.io/tpu-pytorch/xla:r1.13_3.7


COPY requirements-linux.txt /requirements.txt
COPY setup.py /setup.py
COPY environment-linux.yml /environment.yml

RUN python3 -V
RUN python3 -m pip install -r requirements.txt

# RUN conda update -n base conda --yes
# RUN conda init bash
# # SHELL ["/bin/sh", "--login", "-c"]
# RUN . /root/.bashrc && conda env create -f environment.yml
# RUN jupyter kernelspec list --json
# Make RUN commands use the new environment:
# RUN echo "conda activate stable-app" >> ~/.bashrc
# SHELL ["/bin/sh", "--login", "-c"]

# COPY main.py entrypoint.sh conda-init.sh ./
# RUN cat /conda-init.sh >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]

# RUN apt install -y python3.9-dev

# Install pip
# COPY pips.py get-pip.py
# RUN python3 get-pip.py

# # # bleeding-edge nightly CUDA 1.18 to match the image
# # # RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118
# # # MPS
# # RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/mps

# RUN --mount=type=cache,target=/root/.cache \
#   python3 -m pip install -r requirements-linux.txt



# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["tail", "-f", "/dev/null"]
# CMD ["python3", "main.py"]
