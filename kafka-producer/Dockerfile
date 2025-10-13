FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kafka_producer.py .

# Create logs directory
RUN mkdir -p /app/logs/kafka

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "kafka_producer.py"]