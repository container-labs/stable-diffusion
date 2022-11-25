FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN pip install Flask
RUN pip install diffusers
RUN pip install transformers
RUN pip install accelerate
RUN pip install flask-cors

RUN apt-get install -y gcsfuse

RUN mkdir -p /mnt/md-ml
RUN mkdir -p /mnt/md-ml-public

COPY . .

CMD ["python", "main.py"]
