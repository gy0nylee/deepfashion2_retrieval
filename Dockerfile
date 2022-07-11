FROM python:3.9.2-slim
RUN apt-get update
WORKDIR /deepfashion2
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt