from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel
from app.database import get_db
from app.services.ai_service import AIService
from app.services.data_service import DataService
from app.services.scraper_service import MedicalKnowledgeScraper

router = APIRouter()

class SymptomRequest(BaseModel):
    symptoms: List[str]

class DiseaseTreatmentRequest(BaseModel):
    disease_id: int

@router.post("/diagnose", response_model=Dict[str, Any])
def diagnose(request: SymptomRequest, db: Session = Depends(get_db)):
    """
    Generate a diagnosis based on provided symptoms
    """
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
    
    # Use the AI service to diagnose
    diagnosis = AIService.diagnose(db, request.symptoms)
    return diagnosis

@router.post("/treatment", response_model=Dict[str, Any])
def get_treatment(request: DiseaseTreatmentRequest, db: Session = Depends(get_db)):
    """
    Get treatment information for a specific disease
    """
    if not request.disease_id:
        raise HTTPException(status_code=400, detail="No disease ID provided")
    
    treatment = AIService.get_treatment(db, request.disease_id)
    
    if "error" in treatment:
        raise HTTPException(status_code=404, detail=treatment["error"])
    
    return treatment

@router.post("/import-data", response_model=Dict[str, str])
def import_csv_data(db: Session = Depends(get_db)):
    """
    Import data from the CSV file to the database
    """
    try:
        csv_path = "/home/john/Desktop/medical_diagnosis_app/app/data/datasets/Diseases_Symptoms.csv"
        DataService.import_csv_data(db, csv_path)
        return {"message": "Data imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import data: {str(e)}")

@router.post("/scrape-data", response_model=Dict[str, str])
def scrape_data(db: Session = Depends(get_db)):
    """
    Scrape medical data from websites
    """
    try:
        scraper = MedicalKnowledgeScraper(db)
        scraper.run_scraper()
        return {"message": "Data scraped and imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape data: {str(e)}")