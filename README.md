# 🌍 EnviroAudit
**AI-Powered Environmental Compliance Monitoring Platform**

EnviroAudit is an end-to-end, applied AI system designed to automate environmental compliance monitoring using imagery analysis.
The project focuses on real-world deployment concerns such as interpretability, cost-free data access, system robustness, and actionable decision outputs rather than standalone model performance.

It combines computer vision, decision logic, a RESTful backend, and an interactive dashboard to convert raw images and satellite data into explainable compliance risk assessments.

## 🎯 Problem Statement

Environmental compliance monitoring is typically:
- **Manual and time-consuming**
- **Dependent on costly satellite APIs**
- **Difficult to scale and audit**

EnviroAudit addresses this by:
- **Automating visual site analysis using AI**
- **Translating model outputs into clear risk levels and inspection recommendations**
- **Providing a system that is demo-friendly, extensible, and interpretable**

## ✨ Core Capabilities

### 🛰️ Zero-Key Satellite Imagery Pipeline
- **Multi-Provider Orchestration**:
  - **NASA GIBS** (near real-time satellite imagery)
  - **OpenStreetMap** (map tiles)
  - **Synthetic imagery fallback** for resilience
- **No API keys required**, enabling cost-free demos and portfolio deployment
- **Temporal change detection** to identify new construction or environmental changes over time

### 🔍 AI-Driven Visual Analysis
- **Zero-shot image understanding** using **CLIP** for flexible classification
- **Image captioning** via **BLIP** to generate human-readable scene descriptions
- **Object detection** with **GroundingDINO** to identify construction vehicles and heavy machinery
- Designed to **gracefully degrade** when advanced models are unavailable

### ⚠️ Compliance Risk Assessment
- Converts raw ML outputs into **actionable compliance signals**
- Rule-based reasoning layer maps detections to:
  - **Risk levels**: Low, Medium, High, Critical
  - **Inspection recommendations**
- Emphasis on **interpretability and decision support**, not black-box predictions

## 🧩 System Architecture
- **FastAPI backend** serving analysis, location-based monitoring, and change detection
- **Persistent storage** of analysis results for auditability
- **Streamlit dashboard** for visualization, manual uploads, and report generation
- **Modular pipeline design** for easy extension to video feeds, drones, or GIS systems

## 🛠️ Technology Stack

### Backend
- Python, FastAPI, Uvicorn
- SQLAlchemy (SQLite by default, PostgreSQL-ready)

### AI / ML
- PyTorch
- Hugging Face Transformers (CLIP, BLIP)
- GroundingDINO (object detection)

### Frontend
- Streamlit
- Plotly

### Geospatial
- NASA GIBS
- OpenStreetMap
- Folium

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
