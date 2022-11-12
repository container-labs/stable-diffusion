# https://console.cloud.google.com/storage/browser/deeplearning-platform-release/installed-dependencies/containers/m88;tab=objects?_ga=2.49264870.1594978445.1668161991-1121739324.1661533592&pli=1&prefix=&forceOnObjectsSortingFiltering=false
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN mkdir /ml-workspace
WORKDIR /ml-workspace

# among other things, adds add-apt-repository
# used to install newer versions of Python
RUN apt install -y software-properties-common
RUN add-apt-repository -y 'ppa:deadsnakes/ppa'
RUN apt update -y && apt install python3.9 -y
RUN apt install python3.9-dev -y

COPY . .
RUN conda-env create -n training -f environment.yaml
