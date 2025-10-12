# ULTIMATE BULLETPROOF VERSION - Uses Java 21 (available in Debian Trixie)
FROM python:3.11-slim

WORKDIR /app

# Install Java 21 (the ONLY version available in Debian Trixie)
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Java 21
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install PySpark (works perfectly with Java 21)
RUN pip install --no-cache-dir pyspark==3.5.0

# Create directories
RUN mkdir -p /app/logs /app/logs/data /app/checkpoints

# Copy application
COPY spark_consumer.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

CMD ["python", "spark_consumer.py"]