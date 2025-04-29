FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client_app.py /app/

RUN mkdir -p /app/data /app/certificates

ENTRYPOINT ["python", "/app/client_app.py"]