FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV TORCH_HOME=/tmp/torch_cache

# Set working directory
WORKDIR /app

# Install minimal system utilities + OpenCV dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure caching and data paths exist
RUN mkdir -p /tmp/huggingface_cache /tmp/torch_cache /tmp/enviroaudit_data

# Expose the port
EXPOSE 7860

# Startup orchestrator
CMD ["python", "space_startup.py"]