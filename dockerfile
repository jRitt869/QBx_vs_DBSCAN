# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLUSTER_DATA_DIR=/visualize-clusters/output

WORKDIR /visualize-clusters

COPY requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5006

ENTRYPOINT ["sh", "./start.sh"]
