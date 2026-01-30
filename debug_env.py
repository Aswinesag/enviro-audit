from dotenv import load_dotenv
import os
from src.core.config import Settings

print("--- JS Loading .env ---")
load_dotenv()
print(f"SENTINELHUB_CLIENT_ID in os.environ: {'SENTINELHUB_CLIENT_ID' in os.environ}")
print(f"sentinelhub_client_id in os.environ: {'sentinelhub_client_id' in os.environ}")

print("\n--- Pydantic Settings ---")
try:
    settings = Settings()
    print(f"settings.sentinelhub_client_id: {settings.sentinelhub_client_id}")
    print(f"settings.sentinelhub_client_secret: {'*' * 5 if settings.sentinelhub_client_secret else None}")
except Exception as e:
    print(f"Failed to load Settings: {e}")
