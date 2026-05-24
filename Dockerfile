# Dockerfile for Hugging Face Spaces
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV TORCH_HOME=/tmp/torch_cache

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create cache directories
RUN mkdir -p /tmp/huggingface_cache /tmp/torch_cache

# Expose the port
EXPOSE 7860

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]