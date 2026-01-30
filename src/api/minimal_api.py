# minimal_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import uvicorn
from datetime import datetime

# Simple test without full pipeline
app = FastAPI(title="EnviroAudit Test API")

@app.get("/")
async def root():
    return {"message": "EnviroAudit Test API", "status": "running"}

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Simple endpoint to test file upload."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_size": f"{image.width}x{image.height}",
        "processed_at": datetime.now().isoformat(),
        "message": "File received successfully"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": "Not loaded in test API"
    }

if __name__ == "__main__":
    print("Starting minimal test API on http://localhost:8000")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run("minimal_api:app", host="0.0.0.0", port=8000, reload=True)