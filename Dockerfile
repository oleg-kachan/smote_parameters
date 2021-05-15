
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TZ Europe/Moscow

RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y

RUN apt-get -y install python3.9 python3.9-dev python3.9-distutils build-essential curl locales
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

WORKDIR /exp

COPY ./requirements.txt .
RUN pip install numpy Cython && pip install -r requirements.txt

COPY ./data /exp/data
COPY ./wasserstein_smote /exp/wasserstein_smote
COPY ./parameter_search.py /exp/parameter_search.py

ENTRYPOINT ["python3.9", "./parameter_search.py"]