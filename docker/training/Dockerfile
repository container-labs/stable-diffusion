# https://console.cloud.google.com/storage/browser/deeplearning-platform-release/installed-dependencies/containers/m88;tab=objects?_ga=2.49264870.1594978445.1668161991-1121739324.1661533592&pli=1&prefix=&forceOnObjectsSortingFiltering=false
ARG BASE_IMAGE=us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training-base
FROM ${BASE_IMAGE}:latest

RUN pip install cloudml-hypertune

COPY . .
RUN pwd
RUN ls -al

CMD ["./train-style.sh", "--output=output", "--model=CompVis/stable-diffusion-v1-4", "--data=/image_dir"]
# CMD ["tail", "-f", "/dev/null"]
