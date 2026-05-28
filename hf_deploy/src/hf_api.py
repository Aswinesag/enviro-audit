# src/hf_api.py - Embedded API for Hugging Face Spaces
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uuid
from datetime import datetime

# Create FastAPI app
api_app = FastAPI()

@api_app.get("/health")
async def health():
    return {"status": "healthy", "platform": "huggingface_spaces"}

@api_app.post("/analyze")
async def analyze(file: UploadFile = File(...), latitude: float = None, longitude: float = None, project_id: str = None):
    from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline as AnalysisPipeline
    
    # Initialize pipeline (cached)
    if not hasattr(analyze, "pipeline"):
        analyze.pipeline = AnalysisPipeline()
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    metadata = {
        "filename": file.filename,
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "timestamp": datetime.now().isoformat(),
        "location": {"latitude": latitude, "longitude": longitude} if latitude and longitude else None
    }
    
    result = analyze.pipeline.analyze_image(image, metadata)
    return result

def start_api():
    """Start FastAPI in background thread"""
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="warning")

# Start API in background when module loads
api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()