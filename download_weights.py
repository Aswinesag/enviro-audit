import os
import requests

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✓ Download complete")
    else:
        print(f"✗ Failed to download: {response.status_code}")

# Config
config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"

# Weights
# Using a specific release asset or direct link if possible. 
# Re-using the logic typically found in their repo or documentation.
weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
weights_path = "weights/groundingdino_swint_ogc.pth"

if not os.path.exists(config_path):
    download_file(config_url, config_path)
else:
    print(f"Config already exists at {config_path}")

if not os.path.exists(weights_path):
    download_file(weights_url, weights_path)
else:
    print(f"Weights already exists at {weights_path}")
