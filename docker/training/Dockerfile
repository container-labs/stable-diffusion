# https://console.cloud.google.com/storage/browser/deeplearning-platform-release/installed-dependencies/containers/m88;tab=objects?_ga=2.49264870.1594978445.1668161991-1121739324.1661533592&pli=1&prefix=&forceOnObjectsSortingFiltering=false
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN mkdir /workspace
WORKDIR /workspace

# among other things, adds add-apt-repository
# used to install newer versions of Python
RUN apt install -y software-properties-common
RUN add-apt-repository -y 'ppa:deadsnakes/ppa'
RUN apt update -y && apt install python3.9 -y
RUN apt install python3.9-dev -y

COPY requirements-s.txt requirements.txt
COPY setup.py .
# RUN pip uninstall flake8 -y
#RUN python3.9 -m pip install -r requirements.txt

COPY . .

RUN conda env create -n test-conda -f environment.yaml

# to test dependencies
# RUN python3.9 textual_inversion.py

# RUN mkdir -p models/ldm/text2img-large/
# RUN wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

CMD ["tail", "-f", "/dev/null"]
