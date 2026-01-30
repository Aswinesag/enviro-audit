# 🌍 EnviroAudit

**AI-Powered Environmental Compliance Monitoring System**

EnviroAudit is a comprehensive solution currently undergoing active development. It automates environmental compliance monitoring using advanced Artificial Intelligence, combining computer vision for site analysis with a robust backend API and an interactive dashboard.

## ✨ Key Features

-   **Zero-Key Satellite Integration**:
    -   **Multi-Provider Support**: Uses a smart orchestrator to fetch data from **NASA GIBS** (Real-time), **OpenStreetMap** (Map Tiles), or **Synthetic Generation** (Fallback).
    -   **No API Keys Required**: Designed for seamless portfolio demonstration without expensive enterprise keys.
    -   **Change Detection**: Compares imagery over time to detect new construction or environmental changes.
    
-   **🔍 AI-Powered Analysis**:
    -   **CLIP & BLIP**: Zero-shot classification and image captioning for deep context.
    -   **GroundingDINO**: State-of-the-art object detection to identify heavy machinery and construction vehicles.
    
-   **🚀 RESTful API**: High-performance backend built with **FastAPI**.
-   **📊 Interactive Dashboard**: Professional **Streamlit** interface for data visualization, manual image uploads, and report generation.
-   **🛡️ Risk Assessment**: Automated compliance scoring and risk level categorization (Low, Medium, High, Critical).
-   **💾 Database Integration**: Persistent storage of analysis history using SQLAlchemy.

## 🛠️ Technology Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Frontend**: Streamlit, Plotly
-   **AI/ML**: PyTorch, Hugging Face Transformers, Pillow, GroundingDINO
-   **Satellite**: NASA GIBS, OpenStreetMap, Folium
-   **Database**: SQLAlchemy, SQLite (default) / PostgreSQL compatible

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd EnviroAudit
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If GroundingDINO fails to install, the system will gracefully degrade to use standard classification only.*

4.  **Run the Application**
    
    Start the API Server:
    ```bash
    python main.py
    
    Start the Dashboard (in a new terminal):
    ```bash
    streamlit run app.py
    
To install manually:
```bash
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
```

## 📚 API Documentation

Once the API server is running (`python main.py`), check the interactive docs:
-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Key Endpoints
-   `POST /analyze`: Standard image compliance analysis.
-   `POST /analyze-location`: Fetch and analyze satellite imagery for a specific lat/lon.
-   `POST /compare-location`: Compare two dates for a location to detect changes.

## 📂 Project Structure

```
EnviroAudit/
├── app.py                 # Streamlit Dashboard
├── main.py                # API Server Entry
├── src/                   # Source Code
│   ├── api/               # FastAPI Endpoints
│   ├── core/              # Config & Database
│   ├── models/            # AI Models (CLIP, BLIP, GroundingDINO)
│   ├── pipelines/         # Analysis Logic
│   └── utils/
│       └── satellite/     # Zero-Key Satellite Module 🛰️
│           ├── orchestrator.py
│           └── providers/ (NASA, OSM, Synthetic)
└── data/                  # Local storage for results
```

## 🧪 Running Tests
```bash
pytest
```
