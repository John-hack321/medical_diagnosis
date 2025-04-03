from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SymptomRequest(BaseModel):
    symptoms: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "symptoms": ["fever", "cough", "headache"]
            }
        }

class DiagnosisResponse(BaseModel):
    diagnoses: Dict[str, Any]
    symptom_matches: Dict[str, str]

class DiseaseDetail(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    risk_factors: Optional[str] = None
    
    class Config:
        orm_mode = True

class TreatmentRequest(BaseModel):
    disease_id: int
    
    class Config:
        schema_extra = {
            "example": {
                "disease_id": 1
            }
        }

class TreatmentResponse(BaseModel):
    name: str
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    risk_factors: Optional[str] = None