from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.database import get_db
from app.services.ai_service import AIService
from app.services.data_service import DataService
from app.services.scraper_service import MedicalKnowledgeScraper

router = APIRouter()

class SymptomRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms to diagnose", example=["fever", "cough", "headache"])

class NaturalLanguageRequest(BaseModel):
    text: str = Field(..., description="Natural language description of symptoms", 
                     example="I've been having a severe headache, fever, and my body aches all over for the past two days.")
    language: str = Field("english", description="Language of the text (english or swahili)")

class DiseaseTreatmentRequest(BaseModel):
    disease_id: int = Field(..., description="ID of the disease to get treatment for")

class DiseaseInsightRequest(BaseModel):
    disease_id: int = Field(..., description="ID of the disease to get AI insights for")

@router.post("/diagnose", response_model=Dict[str, Any])
def diagnose(request: SymptomRequest, db: Session = Depends(get_db)):
    """
    Generate a diagnosis based on provided symptoms using multiple AI approaches
    """
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
    
    # Use the AI service to diagnose using multiple approaches
    diagnosis = AIService.diagnose(db, request.symptoms)
    return diagnosis

@router.post("/diagnose-text", response_model=Dict[str, Any])
def diagnose_from_text(request: NaturalLanguageRequest, db: Session = Depends(get_db)):
    """
    Extract symptoms from natural language text and generate a diagnosis
    
    This endpoint uses NLP to extract symptoms from the text and then diagnoses based on those symptoms.
    Supports both English and Swahili text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    
    diagnosis = AIService.diagnose_from_text(db, request.text, request.language)
    
    if "error" in diagnosis:
        raise HTTPException(status_code=400, detail=diagnosis["error"])
    
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

@router.post("/treatment-insights", response_model=Dict[str, Any])
def get_treatment_insights(request: DiseaseInsightRequest, db: Session = Depends(get_db)):
    """
    Get enhanced treatment information with AI-generated insights
    """
    if not request.disease_id:
        raise HTTPException(status_code=400, detail="No disease ID provided")
    
    insights = AIService.generate_treatment_insights(db, request.disease_id)
    
    if "error" in insights:
        raise HTTPException(status_code=404, detail=insights["error"])
    
    return insights

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

@router.get("/symptoms", response_model=List[str])
def get_all_symptoms(db: Session = Depends(get_db)):
    """
    Get a list of all known symptoms in the system
    """
    from app.models.disease import Symptom
    symptoms = db.query(Symptom.name).distinct().all()
    return [symptom[0] for symptom in symptoms]

@router.get("/diseases", response_model=List[Dict[str, Any]])
def get_all_diseases(db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    """
    Get a list of all diseases in the system with pagination
    """
    from app.models.disease import Disease
    diseases = db.query(Disease).offset(offset).limit(limit).all()
    
    result = []
    for disease in diseases:
        result.append({
            "id": disease.id,
            "name": disease.name,
            "description": disease.description,
            "symptom_count": len(disease.symptoms)
        })
    
    return result