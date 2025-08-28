# FraudLens Docker Image
# Author: Yobie Benjamin
# Date: 2025-08-26 18:34:00 PDT

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fraudlens/ ./fraudlens/
COPY configs/ ./configs/
COPY data/samples/ ./data/samples/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/plugins /app/data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV FRAUDLENS_ENV=production
ENV FRAUDLENS_MAX_MEMORY_GB=100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fraudlens; print('Health check passed')" || exit 1

# Expose port for API/demo
EXPOSE 8000

# Default command
CMD ["python", "-m", "fraudlens.api.server"]