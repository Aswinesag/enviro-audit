# main.py - Updated for production
import os
import uvicorn
from src.api.endpoints import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "src.api.endpoints:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        workers=1,
        log_level="info"
    )