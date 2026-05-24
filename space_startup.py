# space_startup.py - Entry point for Hugging Face Space
import os
import threading
import time
import streamlit.web.bootstrap as bootstrap

def main():
    print("=" * 60)
    print("🌍 EnviroAudit - Starting on Hugging Face Spaces")
    print("=" * 60)
    print(f"Memory available: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3):.1f} GB")
    print("=" * 60)
    
    # Start API in background
    from src.hf_api import start_api
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    print("✅ API started on port 8000")
    print("⏳ Waiting for API to initialize...")
    time.sleep(5)
    
    # Streamlit will be started by HF Spaces automatically
    print("✅ Ready for requests!")

if __name__ == "__main__":
    main()