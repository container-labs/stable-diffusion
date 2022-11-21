FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN pip install Flask
RUN pip install diffusers
RUN pip install transformers
RUN pip install accelerate

COPY . .

CMD ["python", "main.py"]
