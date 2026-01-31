FROM python:3.10-slim

# Install system dependencies for OpenCV and GroundingDINO
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# Expose Streamlit port
EXPOSE 7860

# Run both the API and Streamlit (Simplified for Space)
CMD python main.py & streamlit run app.py --server.port 7860 --server.address 0.0.0.0