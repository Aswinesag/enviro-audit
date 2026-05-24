# Dockerfile - For Render deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy application
COPY . .

# Create cache directory
RUN mkdir -p /tmp/huggingface_cache

# Download models during build
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/tmp/huggingface_cache')"
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='/tmp/huggingface_cache')"

EXPOSE 8000

CMD ["python", "main.py"]