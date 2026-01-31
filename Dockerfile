# 1. Use an optimized Python image with build tools
FROM python:3.10-slim

# 2. Install system dependencies for OpenCV and GroundingDINO compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Set work directory and copy project files
WORKDIR /app
COPY . .

# 4. Install dependencies from your requirements.txt
# This includes fastapi, transformers, torch, and geospatial tools
RUN pip install --no-cache-dir -r requirements.txt

# 5. Build GroundingDINO from source as required for object detection
RUN pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# 6. Expose the Hugging Face default port
EXPOSE 7860

# 7. Start the FastAPI backend (port 8000) and Streamlit (port 7860)
# This allows the app.py to communicate with localhost:8000
CMD python main.py & streamlit run app.py --server.port 7860 --server.address 0.0.0.0