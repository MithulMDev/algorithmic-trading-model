# Multi-Service OHLCV Processing Pipeline Dockerfile
# Used by: spark-consumer, signal-processor, portfolio-manager
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    procps \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

# Copy and install Python dependencies
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/logs/data /app/logs/signals /app/logs/portfolio /app/checkpoints

# Copy all application files
COPY spark_consumer.py .
COPY influx_manager.py .
COPY clickhouse_manager.py .
COPY indicator_engine.py .
COPY signal_processor.py .
COPY portfolio_manager.py .

# Copy health check scripts
COPY healthcheck.sh .
RUN chmod +x /app/healthcheck.sh

# Default command (can be overridden in docker-compose)
CMD ["python", "spark_consumer.py"]
