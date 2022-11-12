# https://console.cloud.google.com/storage/browser/deeplearning-platform-release/installed-dependencies/containers/m88;tab=objects?_ga=2.49264870.1594978445.1668161991-1121739324.1661533592&pli=1&prefix=&forceOnObjectsSortingFiltering=false
FROM us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/base:latest

COPY . .
RUN pwd
RUN ls -al

RUN conda init bash && \
  eval "$(conda shell.bash hook)" && \
  conda activate training && \
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia

CMD ["tail", "-f", "/dev/null"]
