# src/core/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict, Any

Base = declarative_base()

class AnalysisResult(Base):
    """Database model for storing analysis results."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, unique=True, index=True, nullable=False)
    project_id = Column(String, index=True, nullable=True)
    
    # Image info
    image_size = Column(String, nullable=True)
    image_source = Column(String, nullable=True)
    image_format = Column(String, nullable=True)
    
    # Analysis results
    primary_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    caption = Column(Text, nullable=True)
    
    # Location
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Full results (as JSON)
    full_results = Column(JSON, nullable=True)
    
    # Compliance flags
    requires_inspection = Column(Boolean, default=False, index=True)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False, index=True)
    inspection_notes = Column(Text, nullable=True)
    
    # Metadata
    batch_id = Column(String, nullable=True, index=True)
    processing_time = Column(Float, nullable=True)  # Seconds

class DatabaseManager:
    """Manage database operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            # Default to SQLite for development
            os.makedirs("data", exist_ok=True)
            database_url = "sqlite:///data/enviroaudit.db"
        
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        print(f"Database initialized: {database_url}")
    
    def get_db(self) -> Session:
        """Get database session."""
        db = self.SessionLocal()
        try:
            return db
        finally:
            db.close()
    
    def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Save analysis results to database."""
        db = self.get_db()
        
        try:
            # Extract data
            metadata = analysis_data.get('metadata', {})
            classification = analysis_data.get('classification', {})
            caption = analysis_data.get('caption', {})
            compliance = analysis_data.get('compliance', {})
            image_info = analysis_data.get('image_info', {})
            
            # Create result object
            result = AnalysisResult(
                analysis_id=metadata.get('analysis_id'),
                project_id=metadata.get('project_id'),
                
                image_size=f"{image_info.get('original_size', [0, 0])[0]}x{image_info.get('original_size', [0, 0])[1]}",
                image_source=metadata.get('source_url') or metadata.get('filename'),
                image_format=image_info.get('format'),
                
                primary_label=classification.get('primary_label'),
                confidence=float(classification.get('primary_confidence', '0%').strip('%')) / 100,
                risk_level=compliance.get('risk_level'),
                caption=caption.get('basic_caption'),
                
                latitude=metadata.get('location', {}).get('latitude'),
                longitude=metadata.get('location', {}).get('longitude'),
                
                full_results=analysis_data,
                requires_inspection=compliance.get('risk_level') in ['CRITICAL', 'HIGH'],
                
                analyzed_at=datetime.fromisoformat(analysis_data.get('timestamp')) if analysis_data.get('timestamp') else datetime.utcnow()
            )
            
            db.add(result)
            db.commit()
            db.refresh(result)
            
            print(f"Analysis saved to database: {result.analysis_id}")
            return result.analysis_id
            
        except Exception as e:
            db.rollback()
            print(f"Failed to save analysis: {e}")
            raise
        finally:
            db.close()
    
    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Retrieve analysis by ID."""
        db = self.get_db()
        try:
            return db.query(AnalysisResult).filter(AnalysisResult.analysis_id == analysis_id).first()
        finally:
            db.close()
    
    def get_analyses(
        self,
        project_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        requires_inspection: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[AnalysisResult]:
        """Get analyses with filtering."""
        db = self.get_db()
        try:
            query = db.query(AnalysisResult)
            
            if project_id:
                query = query.filter(AnalysisResult.project_id == project_id)
            if risk_level:
                query = query.filter(AnalysisResult.risk_level == risk_level)
            if requires_inspection is not None:
                query = query.filter(AnalysisResult.requires_inspection == requires_inspection)
            
            return query.order_by(AnalysisResult.analyzed_at.desc()).offset(offset).limit(limit).all()
        finally:
            db.close()
    
    def get_analysis_count(
        self,
        project_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        requires_inspection: Optional[bool] = None
    ) -> int:
        """Get count of analyses with filtering."""
        db = self.get_db()
        try:
            query = db.query(func.count(AnalysisResult.id))
            
            if project_id:
                query = query.filter(AnalysisResult.project_id == project_id)
            if risk_level:
                query = query.filter(AnalysisResult.risk_level == risk_level)
            if requires_inspection is not None:
                query = query.filter(AnalysisResult.requires_inspection == requires_inspection)
            
            return query.scalar()
        finally:
            db.close()
    
    def update_inspection_status(
        self,
        analysis_id: str,
        scheduled: Optional[bool] = None,
        completed: Optional[bool] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update inspection status for an analysis."""
        db = self.get_db()
        try:
            analysis = db.query(AnalysisResult).filter(AnalysisResult.analysis_id == analysis_id).first()
            if not analysis:
                return False
            
            if scheduled is not None:
                analysis.inspection_scheduled = scheduled
            if completed is not None:
                analysis.inspection_completed = completed
            if notes is not None:
                analysis.inspection_notes = notes
            
            analysis.updated_at = datetime.utcnow()
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Failed to update inspection status: {e}")
            raise
        finally:
            db.close()
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get system statistics."""
        db = self.get_db()
        try:
            # Date cutoff
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Total analyses
            total = db.query(func.count(AnalysisResult.id)).scalar() or 0
            
            # Recent analyses
            recent = db.query(func.count(AnalysisResult.id)).filter(
                AnalysisResult.analyzed_at >= cutoff_date
            ).scalar() or 0
            
            # Risk distribution
            risk_distribution = {}
            risk_counts = db.query(
                AnalysisResult.risk_level,
                func.count(AnalysisResult.id)
            ).group_by(AnalysisResult.risk_level).all()
            
            for risk, count in risk_counts:
                risk_distribution[risk] = count
            
            # Projects
            project_count = db.query(func.count(func.distinct(AnalysisResult.project_id))).scalar() or 0
            
            # High risk analyses needing inspection
            high_risk_pending = db.query(func.count(AnalysisResult.id)).filter(
                AnalysisResult.requires_inspection == True,
                AnalysisResult.inspection_completed == False,
                AnalysisResult.analyzed_at >= cutoff_date
            ).scalar() or 0
            
            # Average confidence
            avg_confidence = db.query(func.avg(AnalysisResult.confidence)).scalar() or 0
            
            return {
                "total_analyses": total,
                "recent_analyses": recent,
                "days_considered": days,
                "risk_distribution": risk_distribution,
                "project_count": project_count,
                "high_risk_pending": high_risk_pending,
                "average_confidence": float(avg_confidence),
                "generated_at": datetime.utcnow().isoformat()
            }
        finally:
            db.close()