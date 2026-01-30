# main.py
import uvicorn
from src.api.endpoints import app

if __name__ == "__main__":
    print("="*60)
    print("Starting EnviroAudit API")
    print("="*60)
    print("Models are loading... (this may take a minute)")
    print("Once loaded, visit:")
    print("  - http://localhost:8000/docs for API documentation")
    print("  - http://localhost:8000/ for API info")
    print("="*60)
    
    # Alternative: Don't use reload
    uvicorn.run(
        app,  # Pass the app object directly
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set reload to False
        log_level="info"
    )