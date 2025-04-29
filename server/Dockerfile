FROM python:3.9-slim

WORKDIR /app

# Install necessary system libraries for LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flower
RUN pip install --no-cache-dir flwr==1.5.0 prometheus-client==0.16.0

# Copy the code
COPY server /app/server

# Create directories for data, certificates, and metrics
RUN mkdir -p /app/data /app/certificates /app/metrics

EXPOSE 8080
EXPOSE 8000

# Run server directly instead of using keep_alive
ENTRYPOINT ["python", "/app/server/server_app.py"]