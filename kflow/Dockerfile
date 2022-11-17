FROM python:3.8

RUN pip install google-cloud-aiplatform
RUN pip install google-cloud-pipeline-components
RUN pip install kfp

COPY . .

RUN mkdir /os-shared

CMD ["python", "main.py"]
