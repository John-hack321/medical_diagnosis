from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.database import get_db
from app.services.dialogue_service import DialogueService
from app.services.ai_service import AIService

router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
    responses={404: {"description": "Not found"}}
)

class MessageRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User's message")
    language: str = Field("english", description="Language of the message (english or swahili)")

class DiagnosisRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")

class EndConversationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")

@router.post("/message", response_model=Dict[str, Any])
def process_message(request: MessageRequest, db: Session = Depends(get_db)):
    """
    Process a message in a diagnostic conversation and generate appropriate responses
    
    This endpoint processes a user message in the context of an ongoing diagnostic conversation.
    It extracts symptoms, checks for emergencies, and generates follow-up questions.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = DialogueService.process_message(db, request.user_id, request.message, request.language)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/diagnose", response_model=Dict[str, Any])
def get_diagnosis_from_conversation(request: DiagnosisRequest, db: Session = Depends(get_db)):
    """
    Generate a diagnosis based on the current conversation state
    
    This endpoint uses the symptoms collected during the conversation to generate a diagnosis.
    """
    try:
        # Get conversation state and use it for diagnosis
        diagnosis_result = DialogueService.get_diagnosis(db, request.user_id)
        
        # If we have confirmed symptoms, use the AI service for diagnosis
        if "confirmed_symptoms" in diagnosis_result and diagnosis_result["confirmed_symptoms"]:
            confirmed_symptoms = diagnosis_result["confirmed_symptoms"]
            ai_diagnosis = AIService.diagnose(db, confirmed_symptoms)
            
            # Combine the conversation state with the AI diagnosis
            diagnosis_result["diagnosis"] = ai_diagnosis
        else:
            # Not enough symptoms for a diagnosis
            diagnosis_result["diagnosis"] = {
                "error": "Not enough symptoms collected for a diagnosis",
                "recommendation": "Please provide more information about your symptoms"
            }
        
        return diagnosis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diagnosis: {str(e)}")

@router.post("/end", response_model=Dict[str, str])
def end_conversation(request: EndConversationRequest, db: Session = Depends(get_db)):
    """
    End a diagnostic conversation
    
    This endpoint closes the conversation and cleans up any associated resources.
    """
    try:
        DialogueService.end_conversation(db, request.user_id)
        return {"message": "Conversation ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")