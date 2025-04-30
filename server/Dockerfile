FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir flwr==1.5.0 prometheus-client==0.16.0

COPY server /app/server

RUN mkdir -p /app/data /app/certificates /app/metrics

EXPOSE 8080
EXPOSE 8000

ENTRYPOINT ["python", "/app/server/server_app.py"]