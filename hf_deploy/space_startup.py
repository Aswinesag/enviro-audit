# space_startup.py - Corrected Entry Point for Hugging Face Spaces
import os
import threading
import time

def main():
    print("=" * 60)
    print("🌍 EnviroAudit - Starting on Hugging Face Spaces")
    print("=" * 60)
    
    # 1. Start FastAPI API in background thread
    print("🚀 Initializing FastAPI Backend...")
    from src.hf_api import start_api
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    print("✅ API started on port 8000")
    print("⏳ Waiting for API to initialize...")
    time.sleep(5)
    
    # 2. Start Streamlit in the foreground (blocks and serves on port 7860)
    print("🚀 Starting Streamlit Dashboard...")
    os.system("streamlit run app.py --server.port=7860 --server.address=0.0.0.0")

if __name__ == "__main__":
    main()