# https://console.cloud.google.com/storage/browser/deeplearning-platform-release/installed-dependencies/containers/m88;tab=objects?_ga=2.49264870.1594978445.1668161991-1121739324.1661533592&pli=1&prefix=&forceOnObjectsSortingFiltering=false
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN mkdir /ml-workspace
WORKDIR /ml-workspace

COPY . .
RUN conda init bash
RUN conda-env create -n training -f environment-base.yaml

CMD ["tail", "-f", "/dev/null"]
