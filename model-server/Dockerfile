FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN pip install \
  Flask flask-cors \
  diffusers transformers accelerate

RUN apt-get install -y gcsfuse

RUN mkdir -p /mnt/md-ml
RUN mkdir -p /mnt/md-ml-public

COPY . .

CMD ["python", "main.py"]
