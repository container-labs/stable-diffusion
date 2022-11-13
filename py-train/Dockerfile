FROM python:3.8

RUN pip install google-cloud-aiplatform
RUN pip install google-cloud-pubsub

COPY . .

CMD ["python", "main.py"]
