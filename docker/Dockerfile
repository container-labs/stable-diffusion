FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN mkdir /ml-workspace
WORKDIR /ml-workspace

COPY requirements-linux.txt /ml-workspace/requirements.txt
COPY setup.py /ml-workspace/setup.py
# among other things, adds add-apt-repository
# used to install newer versions of Python
# RUN apt install -y software-properties-common

RUN pip uninstall flake8 -y
RUN pip install -r requirements.txt
ENV XRT_TPU_CONFIG localservice;0;localhost:51011

COPY main.py .

# ENTRYPOINT ["./entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
