# FROM us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest
# FROM python:3.9
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN pip install Flask
RUN pip install diffusers
RUN pip install transformers
RUN pip install accelerate
# RUN pip install pytorch torchvision cudatoolkit==10.1

COPY . .

CMD ["./server.sh"]
