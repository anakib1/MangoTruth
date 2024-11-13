FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./compute ./compute
COPY ./detectors ./detectors
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./compute/requirements.txt
RUN pip install --no-cache-dir -r ./detectors/requirements.txt


CMD ["python", "compute/main.py"]
