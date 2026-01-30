# src/api/endpoints.py - Complete FastAPI Backend
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
from PIL import Image
import io
import uuid
import json
import os
from datetime import datetime
import logging
import traceback

# Import project modules
from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline as AnalysisPipeline
from src.core.config import settings
from src.core.database import DatabaseManager, AnalysisResult
from pydantic import BaseModel

class LocationAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    date: Optional[str] = None
    bbox_size: float = 0.01
    project_id: Optional[str] = "location-analysis"

class LocationComparisonRequest(BaseModel):
    latitude: float
    longitude: float
    date1: str
    date2: Optional[str] = None
    bbox_size: float = 0.01
    project_id: Optional[str] = "change-detection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EnviroAudit API",
    description="AI-Powered Environmental Compliance Monitoring System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pipeline = None
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global pipeline, db_manager
    
    logger.info("Starting EnviroAudit API...")
    
    try:
        # Initialize analysis pipeline
        pipeline = AnalysisPipeline()
        logger.info("✓ Analysis pipeline initialized")
        
        # Initialize database
        db_manager = DatabaseManager()
        logger.info("✓ Database initialized")
        
        # Create data directory if it doesn't exist
        os.makedirs("data/uploads", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        
        logger.info(f"✓ EnviroAudit API v2.0.0 ready on http://{settings.api_host}:{settings.api_port}")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down EnviroAudit API...")

# Root endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "EnviroAudit",
        "version": "2.0.0",
        "description": "AI-powered environmental compliance monitoring",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /analyze": "Analyze uploaded image",
            "POST /analyze-url": "Analyze image from URL",
            "GET /analyses": "Get analysis history",
            "GET /analyses/{analysis_id}": "Get specific analysis",
            "GET /statistics": "Get system statistics"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check pipeline status
        pipeline_status = pipeline is not None
        
        # Check database status
        db_status = False
        if db_manager:
            try:
                # Try a simple query
                count = db_manager.get_analysis_count()
                db_status = True
            except:
                db_status = False
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "analysis_pipeline": pipeline_status,
                "database": db_status
            },
            "system": {
                "python_version": os.sys.version,
                "platform": os.sys.platform
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Analysis endpoints
@app.post("/analyze", tags=["Analysis"], response_model=Dict[str, Any])
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    latitude: Optional[float] = Query(None, description="Latitude coordinate"),
    longitude: Optional[float] = Query(None, description="Longitude coordinate"),
    date: Optional[str] = Query(None, description="Date of image (ISO format)"),
    project_id: Optional[str] = Query("default-project", description="Project identifier"),
    save_to_db: bool = Query(True, description="Save results to database")
):
    """
    Analyze uploaded image for environmental compliance.
    
    Supports: JPEG, PNG, BMP, TIFF images.
    Maximum file size: 10MB (configurable).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, BMP, TIFF)")
    
    try:
        # Read and validate image
        contents = await file.read()
        
        # Check file size (10MB limit)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(contents),
            "upload_time": datetime.now().isoformat(),
            "location": {
                "latitude": latitude,
                "longitude": longitude
            } if latitude and longitude else None,
            "date": date,
            "project_id": project_id,
            "analysis_id": str(uuid.uuid4())
        }
        
        logger.info(f"Analyzing image: {file.filename} ({image.size}) for project: {project_id}")
        
        # Analyze image
        result = pipeline.analyze_image(image, metadata)
        
        # Save to database if requested
        if save_to_db and db_manager:
            try:
                db_id = db_manager.save_analysis(result)
                result["database_id"] = db_id
                result["database_saved"] = True
                logger.info(f"Analysis saved to database: {db_id}")
            except Exception as db_error:
                logger.error(f"Failed to save to database: {db_error}")
                result["database_saved"] = False
                result["database_error"] = str(db_error)
        else:
            result["database_saved"] = False
        
        # Save result to file (for backup)
        result_file = f"data/results/{metadata['analysis_id']}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        # Save image (thumbnail)
        image.thumbnail((300, 300))
        image.save(f"data/uploads/{metadata['analysis_id']}_thumb.jpg", "JPEG")
        
        logger.info(f"Analysis completed: {metadata['analysis_id']}")
        
        return JSONResponse(content=result)
        
    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-url", tags=["Analysis"], response_model=Dict[str, Any])
async def analyze_image_url(
    image_url: str = Query(..., description="URL of image to analyze"),
    latitude: Optional[float] = Query(None, description="Latitude coordinate"),
    longitude: Optional[float] = Query(None, description="Longitude coordinate"),
    date: Optional[str] = Query(None, description="Date of image (ISO format)"),
    project_id: Optional[str] = Query("default-project", description="Project identifier"),
    save_to_db: bool = Query(True, description="Save results to database")
):
    """Analyze image from URL."""
    try:
        import requests
        from io import BytesIO
        
        logger.info(f"Downloading image from URL: {image_url}")
        
        # Download image with timeout
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to a valid image")
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Prepare metadata
        metadata = {
            "source_url": image_url,
            "content_type": content_type,
            "download_time": datetime.now().isoformat(),
            "location": {
                "latitude": latitude,
                "longitude": longitude
            } if latitude and longitude else None,
            "date": date,
            "project_id": project_id,
            "analysis_id": str(uuid.uuid4())
        }
        
        logger.info(f"Analyzing image from URL: {image_url} ({image.size})")
        
        # Analyze image
        result = pipeline.analyze_image(image, metadata)
        
        # Save to database if requested
        if save_to_db and db_manager:
            try:
                db_id = db_manager.save_analysis(result)
                result["database_id"] = db_id
                result["database_saved"] = True
            except Exception as db_error:
                logger.error(f"Failed to save to database: {db_error}")
                result["database_saved"] = False
                result["database_error"] = str(db_error)
        else:
            result["database_saved"] = False
        
        # Save result to file
        result_file = f"data/results/{metadata['analysis_id']}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"URL analysis completed: {metadata['analysis_id']}")
        
        return JSONResponse(content=result)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"URL analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Database endpoints
@app.get("/analyses", tags=["Database"])
async def get_analyses(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    requires_inspection: Optional[bool] = Query(None, description="Filter by inspection requirement")
):
    """Get analysis history with filtering."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        analyses = db_manager.get_analyses(
            project_id=project_id,
            risk_level=risk_level,
            requires_inspection=requires_inspection,
            limit=limit,
            offset=offset
        )
        
        return {
            "count": len(analyses),
            "total": db_manager.get_analysis_count(
                project_id=project_id,
                risk_level=risk_level,
                requires_inspection=requires_inspection
            ),
            "limit": limit,
            "offset": offset,
            "analyses": [
                {
                    "analysis_id": a.analysis_id,
                    "project_id": a.project_id,
                    "primary_label": a.primary_label,
                    "confidence": a.confidence,
                    "risk_level": a.risk_level,
                    "caption": a.caption,
                    "latitude": a.latitude,
                    "longitude": a.longitude,
                    "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else None,
                    "requires_inspection": a.requires_inspection,
                    "inspection_scheduled": a.inspection_scheduled,
                    "inspection_completed": a.inspection_completed,
                    "image_size": a.image_size
                }
                for a in analyses
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch analyses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses: {str(e)}")

@app.get("/analyses/{analysis_id}", tags=["Database"])
async def get_analysis(analysis_id: str):
    """Get specific analysis by ID."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        analysis = db_manager.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
        
        # Return full results if available
        if analysis.full_results:
            return JSONResponse(content=analysis.full_results)
        else:
            return {
                "analysis_id": analysis.analysis_id,
                "project_id": analysis.project_id,
                "primary_label": analysis.primary_label,
                "confidence": analysis.confidence,
                "risk_level": analysis.risk_level,
                "caption": analysis.caption,
                "analyzed_at": analysis.analyzed_at.isoformat() if analysis.analyzed_at else None,
                "requires_inspection": analysis.requires_inspection
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

@app.put("/analyses/{analysis_id}/inspection", tags=["Database"])
async def update_inspection_status(
    analysis_id: str,
    scheduled: Optional[bool] = Query(None, description="Set inspection scheduled status"),
    completed: Optional[bool] = Query(None, description="Set inspection completed status"),
    notes: Optional[str] = Query(None, description="Inspection notes")
):
    """Update inspection status for an analysis."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        success = db_manager.update_inspection_status(
            analysis_id=analysis_id,
            scheduled=scheduled,
            completed=completed,
            notes=notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
        
        return {
            "status": "success",
            "message": f"Inspection status updated for {analysis_id}",
            "analysis_id": analysis_id,
            "updates": {
                "scheduled": scheduled,
                "completed": completed,
                "notes_updated": notes is not None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update inspection status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update inspection status: {str(e)}")

# Statistics endpoint
@app.get("/statistics", tags=["Statistics"])
async def get_statistics(
    days: int = Query(7, description="Number of days to include", ge=1, le=365)
):
    """Get system statistics."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = db_manager.get_statistics(days=days)
        
        # Add pipeline info
        stats["pipeline"] = {
            "status": "active" if pipeline else "inactive",
            "models_loaded": True if pipeline else False
        }
        
        # Add system info
        import psutil
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# Batch operations
@app.post("/batch-analyze", tags=["Batch Operations"])
async def batch_analyze(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    project_id: str = Query(..., description="Project identifier"),
    background_tasks: BackgroundTasks = None
):
    """Analyze multiple images in batch (asynchronous)."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
    
    # Create batch ID
    batch_id = str(uuid.uuid4())
    
    # Store batch info (in production, use database/redis)
    batch_info = {
        "batch_id": batch_id,
        "project_id": project_id,
        "total_files": len(files),
        "processed_files": 0,
        "completed_files": 0,
        "failed_files": 0,
        "start_time": datetime.now().isoformat(),
        "status": "processing",
        "results": []
    }
    
    # Save batch info
    batch_file = f"data/batches/{batch_id}.json"
    os.makedirs("data/batches", exist_ok=True)
    with open(batch_file, "w") as f:
        json.dump(batch_info, f, indent=2)
    
    # Process in background if requested
    if background_tasks:
        background_tasks.add_task(process_batch, batch_id, files, project_id)
        return {
            "batch_id": batch_id,
            "status": "processing",
            "message": f"Batch processing started for {len(files)} files",
            "monitor_url": f"/batch/{batch_id}/status"
        }
    else:
        # Process synchronously
        results = []
        for file in files:
            try:
                # Process each file
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                
                metadata = {
                    "filename": file.filename,
                    "project_id": project_id,
                    "analysis_id": str(uuid.uuid4()),
                    "batch_id": batch_id
                }
                
                result = pipeline.analyze_image(image, metadata)
                results.append({
                    "filename": file.filename,
                    "analysis_id": metadata["analysis_id"],
                    "success": True,
                    "result": result
                })
                
                batch_info["completed_files"] += 1
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                batch_info["failed_files"] += 1
            
            batch_info["processed_files"] += 1
        
        # Update batch info
        batch_info["status"] = "completed"
        batch_info["end_time"] = datetime.now().isoformat()
        batch_info["results"] = results
        
        with open(batch_file, "w") as f:
            json.dump(batch_info, f, indent=2)
        
        return {
            "batch_id": batch_id,
            "status": "completed",
            "total_files": batch_info["total_files"],
            "completed_files": batch_info["completed_files"],
            "failed_files": batch_info["failed_files"],
            "results": results
        }

@app.get("/batch/{batch_id}/status", tags=["Batch Operations"])
async def get_batch_status(batch_id: str):
    """Get status of a batch operation."""
    batch_file = f"data/batches/{batch_id}.json"
    
    if not os.path.exists(batch_file):
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    
    with open(batch_file, "r") as f:
        batch_info = json.load(f)
    
    return batch_info

# Utility functions
async def process_batch(batch_id: str, files: List[UploadFile], project_id: str):
    """Background task to process batch."""
    # Implementation for background processing
    pass

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.post("/analyze-enhanced", tags=["Enhanced Analysis"])
async def analyze_enhanced(
    file: UploadFile = File(...),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    project_id: Optional[str] = Query("default-project"),
    use_detection: bool = Query(True, description="Use GroundingDINO for object detection")
):
    """Enhanced image analysis with object detection."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        metadata = {
            "filename": file.filename,
            "project_id": project_id,
            "analysis_id": str(uuid.uuid4()),
            "analysis_type": "enhanced",
            "use_detection": use_detection
        }
        
        if latitude and longitude:
            metadata["location"] = {"latitude": latitude, "longitude": longitude}
        
        # Use enhanced pipeline
        from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline
        pipeline = EnhancedAnalysisPipeline()
        
        result = pipeline.analyze_image(image, metadata)
        
        # Save to database
        if db_manager:
            try:
                db_manager.save_analysis(result)
                result["database_saved"] = True
            except Exception as db_error:
                logger.error(f"Database save failed: {db_error}")
                result["database_saved"] = False
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/analyze-location", tags=["Satellite Analysis"])
async def analyze_location(
    request: LocationAnalysisRequest
):
    """Analyze a geographic location using satellite imagery."""
    try:
        metadata = {
            "analysis_id": str(uuid.uuid4()),
            "project_id": request.project_id,
            "analysis_type": "satellite_location",
            "timestamp": datetime.now().isoformat()
        }
        
        # Use enhanced pipeline
        from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline
        pipeline = EnhancedAnalysisPipeline()
        
        result = pipeline.analyze_location(
            latitude=request.latitude,
            longitude=request.longitude,
            date=request.date,
            bbox_size=request.bbox_size,
            metadata=metadata
        )
        
        # Save to database
        if db_manager:
            try:
                db_manager.save_analysis(result)
                result["database_saved"] = True
            except Exception as db_error:
                logger.error(f"Database save failed: {db_error}")
                result["database_saved"] = False
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location analysis failed: {str(e)}")

@app.post("/compare-location", tags=["Satellite Analysis"])
async def compare_location(
    request: LocationComparisonRequest
):
    """Compare changes at a location over time."""
    try:
        metadata = {
            "analysis_id": str(uuid.uuid4()),
            "project_id": request.project_id,
            "analysis_type": "change_detection",
            "timestamp": datetime.now().isoformat()
        }
        
        # Use enhanced pipeline
        from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline
        pipeline = EnhancedAnalysisPipeline()
        
        result = pipeline.compare_locations_over_time(
            latitude=request.latitude,
            longitude=request.longitude,
            date1=request.date1,
            date2=request.date2,
            bbox_size=request.bbox_size
        )
        
        # Save to database
        if db_manager and "error" not in result:
            try:
                db_manager.save_analysis(result)
                result["database_saved"] = True
            except Exception as db_error:
                logger.error(f"Database save failed: {db_error}")
                result["database_saved"] = False
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change detection failed: {str(e)}")

@app.get("/system-capabilities", tags=["System Info"])
async def get_system_capabilities():
    """Get information about system capabilities and models."""
    try:
        # Check GroundingDINO availability
        from src.models.grounding_dino_detector import GroundingDINODetector
        detector = GroundingDINODetector()
        
        # Check satellite client
        from src.utils.satellite_client import SatelliteImageryClient
        satellite_client = SatelliteImageryClient()
        
        return {
            "models": {
                "clip_classifier": True,
                "blip_captioner": True,
                "grounding_dino": detector.is_available(),
                "satellite_imagery": True
            },
            "features": {
                "image_analysis": True,
                "object_detection": detector.is_available(),
                "satellite_analysis": True,
                "change_detection": True,
                "location_analysis": True
            },
            "pipeline_versions": {
                "standard": "1.0",
                "enhanced": "2.0"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "models": {
                "clip_classifier": True,
                "blip_captioner": True,
                "grounding_dino": False,
                "satellite_imagery": True
            }
        }

# Mount static files (for reports, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# For development/testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.endpoints:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info"
    )