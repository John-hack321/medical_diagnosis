import logging
import re
import json
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DialogueState:
    """Class to represent the current state of a diagnostic conversation"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_id = f"conv_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.confirmed_symptoms = set()
        self.denied_symptoms = set()
        self.uncertain_symptoms = set()
        self.asked_symptoms = set()
        self.conversation_history = []
        self.current_diagnoses = []
        self.emergency_flags = []
        self.language = "english"
        self.last_interaction = datetime.now()
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history"""
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self.last_interaction = datetime.now()
    
    def add_system_message(self, message: str, message_type: str = "general") -> None:
        """Add a system message to the conversation history"""
        self.conversation_history.append({
            "role": "system",
            "message": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        })
        self.last_interaction = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dialogue state to a dictionary for storage"""
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "confirmed_symptoms": list(self.confirmed_symptoms),
            "denied_symptoms": list(self.denied_symptoms),
            "uncertain_symptoms": list(self.uncertain_symptoms),
            "asked_symptoms": list(self.asked_symptoms),
            "conversation_history": self.conversation_history,
            "current_diagnoses": self.current_diagnoses,
            "emergency_flags": self.emergency_flags,
            "language": self.language,
            "last_interaction": self.last_interaction.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueState':
        """Create a DialogueState object from a dictionary"""
        state = cls(data["user_id"])
        state.conversation_id = data["conversation_id"]
        state.confirmed_symptoms = set(data["confirmed_symptoms"])
        state.denied_symptoms = set(data["denied_symptoms"])
        state.uncertain_symptoms = set(data["uncertain_symptoms"])
        state.asked_symptoms = set(data["asked_symptoms"])
        state.conversation_history = data["conversation_history"]
        state.current_diagnoses = data["current_diagnoses"]
        state.emergency_flags = data["emergency_flags"]
        state.language = data["language"]
        state.last_interaction = datetime.fromisoformat(data["last_interaction"])
        return state


class DialogueManager:
    """Manager for diagnostic conversations with users"""
    
    # Emergency symptoms that require immediate attention
    EMERGENCY_SYMPTOMS = {
        "chest pain": "Severe chest pain can be a sign of a heart attack or other serious conditions.",
        "difficulty breathing": "Severe breathing difficulty can indicate a serious respiratory condition or allergic reaction.",
        "sudden severe headache": "A sudden, severe headache can be a sign of a stroke or aneurysm.",
        "sudden numbness": "Sudden numbness, especially on one side of the body, can indicate a stroke.",
        "severe abdominal pain": "Severe, sudden abdominal pain can indicate appendicitis or other emergencies.",
        "coughing up blood": "Coughing up blood can indicate a serious lung or throat condition.",
        "severe vomiting": "Severe, persistent vomiting can lead to dehydration and may indicate serious conditions.",
        "loss of consciousness": "Loss of consciousness requires immediate medical attention.",
        "seizure": "Seizures, especially without a previous diagnosis, require emergency care."
    }
    
    def __init__(self, db: Session):
        self.db = db
        self.sessions = {}  # Store active dialogue sessions
        self.symptom_followups = self._load_symptom_followups()
        self.common_symptoms = self._load_common_symptoms()
        self.all_symptoms = set()
        self._load_all_symptoms()
    
    def _load_all_symptoms(self) -> None:
        """Load all symptoms from the database"""
        try:
            symptoms = self.db.query(Symptom.name).all()
            self.all_symptoms = {s[0].lower() for s in symptoms}
            logger.info(f"Loaded {len(self.all_symptoms)} symptoms from database")
        except Exception as e:
            logger.error(f"Error loading symptoms: {str(e)}")
            self.all_symptoms = set()
    
    def _load_symptom_followups(self) -> Dict[str, List[str]]:
        """Load follow-up questions for specific symptoms"""
        # In a real implementation, this would be loaded from a database or file
        return {
            "fever": [
                "How high is your fever?", 
                "How long have you had the fever?",
                "Does it come and go, or is it constant?"
            ],
            "headache": [
                "Where exactly is the pain located?",
                "How would you rate the pain on a scale of 1-10?",
                "Does the headache get worse with movement or certain activities?"
            ],
            "cough": [
                "Is your cough dry, or are you coughing up mucus?",
                "How long have you been coughing?",
                "Is the cough worse at any particular time of day?"
            ],
            "rash": [
                "Where is the rash located?",
                "Is the rash itchy or painful?",
                "Have you used any new products recently (soaps, detergents, etc.)?"
            ],
            "abdominal pain": [
                "Where exactly is the pain?",
                "Is the pain constant or does it come and go?",
                "Does eating make it better or worse?"
            ],
            "diarrhea": [
                "How many times a day are you experiencing diarrhea?",
                "Is there any blood in your stool?",
                "Have you recently consumed any new foods or drinks?"
            ]
        }
    
    def _load_common_symptoms(self) -> List[str]:
        """Load common symptoms to ask about if no specific leads"""
        # In a real implementation, this would be loaded from a database based on prevalence
        return [
            "fever", "headache", "cough", "sore throat", "fatigue", 
            "body aches", "nausea", "diarrhea", "vomiting", 
            "shortness of breath", "rash", "chest pain"
        ]
    
    def get_session(self, user_id: str) -> DialogueState:
        """Get or create a dialogue session for a user"""
        if user_id not in self.sessions:
            self.sessions[user_id] = DialogueState(user_id)
        return self.sessions[user_id]
    
    def save_session(self, state: DialogueState) -> None:
        """Save the current dialogue state"""
        self.sessions[state.user_id] = state
        # In a real implementation, you would also persist this to a database
    
    def process_message(self, user_id: str, message: str, language: str = "english") -> Dict[str, Any]:
        """
        Process a message from the user in a diagnostic conversation
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message
            language: The language of the message
            
        Returns:
            Response including the next action, follow-up questions, etc.
        """
        # Get or create session
        session = self.get_session(user_id)
        session.language = language
        session.add_user_message(message)
        
        # Check for emergency symptoms in the message
        emergency_check = self._check_emergency_symptoms(message)
        if emergency_check["emergency_detected"]:
            session.emergency_flags.append(emergency_check)
            session.add_system_message(
                emergency_check["response"], 
                message_type="emergency"
            )
            self.save_session(session)
            return {
                "message_type": "emergency",
                "response": emergency_check["response"],
                "emergency_info": emergency_check
            }
        
        # Extract symptoms from the message
        extracted_symptoms = self._extract_symptoms_from_text(message)
        for symptom in extracted_symptoms:
            if symptom not in session.confirmed_symptoms and symptom not in session.denied_symptoms:
                session.confirmed_symptoms.add(symptom)
        
        # Update session with extracted symptoms
        self.save_session(session)
        
        # Generate follow-up questions based on the current state
        followups = self._generate_followup_questions(session)
        
        # Add the followup to the session history
        for question in followups["questions"]:
            session.add_system_message(question, message_type="followup")
            session.asked_symptoms.add(followups["target_symptom"])
        
        # Save the updated session
        self.save_session(session)
        
        return {
            "message_type": "followup",
            "extracted_symptoms": list(extracted_symptoms),
            "confirmed_symptoms": list(session.confirmed_symptoms),
            "followup_questions": followups["questions"],
            "target_symptom": followups["target_symptom"],
            "conversation_id": session.conversation_id
        }
    
    def _check_emergency_symptoms(self, message: str) -> Dict[str, Any]:
        """Check if the message contains any emergency symptoms"""
        message_lower = message.lower()
        
        for symptom, warning in self.EMERGENCY_SYMPTOMS.items():
            if symptom in message_lower:
                return {
                    "emergency_detected": True,
                    "symptom": symptom,
                    "response": f"WARNING: {warning} Please seek immediate medical attention or call emergency services."
                }
        
        return {"emergency_detected": False}
    
    def _extract_symptoms_from_text(self, text: str) -> Set[str]:
        """Extract symptoms from text using simple pattern matching"""
        # In a real implementation, this would use the NLP service
        # Here we're using a simplified version
        
        extracted = set()
        text_lower = text.lower()
        
        # Check for each known symptom
        for symptom in self.all_symptoms:
            # For multi-word symptoms, require exact match
            if " " in symptom and symptom in text_lower:
                extracted.add(symptom)
            # For single-word symptoms, require word boundaries
            elif " " not in symptom and re.search(r'\b' + re.escape(symptom) + r'\b', text_lower):
                extracted.add(symptom)
        
        return extracted
    
    def _generate_followup_questions(self, session: DialogueState) -> Dict[str, Any]:
        """Generate appropriate follow-up questions based on the conversation state"""
        # If we have confirmed symptoms, ask follow-up questions about them
        for symptom in session.confirmed_symptoms:
            if symptom in self.symptom_followups and symptom not in session.asked_symptoms:
                return {
                    "questions": self.symptom_followups[symptom],
                    "target_symptom": symptom
                }
        
        # If we've run out of specific follow-ups, ask about common symptoms
        for symptom in self.common_symptoms:
            if symptom not in session.confirmed_symptoms and symptom not in session.denied_symptoms and symptom not in session.asked_symptoms:
                question = f"Have you been experiencing {symptom}?"
                return {
                    "questions": [question],
                    "target_symptom": symptom
                }
        
        # If we've asked about all common symptoms, give a general response
        return {
            "questions": ["Is there anything else about your symptoms you'd like to share?"],
            "target_symptom": "general"
        }
    
    def get_diagnosis_from_session(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Generate a diagnosis based on the current session state
        
        Args:
            user_id: The user's ID
            
        Returns:
            List of potential diagnoses based on confirmed symptoms
        """
        # This method would integrate with the AI service to get actual diagnoses
        # For now, we'll just return the confirmed symptoms
        session = self.get_session(user_id)
        return {
            "confirmed_symptoms": list(session.confirmed_symptoms),
            "session_state": session.to_dict()
        }
    
    def close_session(self, user_id: str) -> None:
        """Close and save a user's session"""
        if user_id in self.sessions:
            # In a real implementation, you would persist the session to a database
            del self.sessions[user_id]
            logger.info(f"Closed session for user {user_id}")


class DialogueService:
    """Service interface for dialogue functionality"""
    
    # Cache the manager instance
    _manager_instance = None
    
    @classmethod
    def _get_manager(cls, db: Session) -> DialogueManager:
        """Get or create a DialogueManager instance"""
        if cls._manager_instance is None:
            cls._manager_instance = DialogueManager(db)
        return cls._manager_instance
    
    @staticmethod
    def process_message(db: Session, user_id: str, message: str, language: str = "english") -> Dict[str, Any]:
        """
        Process a message in a diagnostic conversation
        
        Args:
            db: Database session
            user_id: User identifier
            message: User's message
            language: Message language
            
        Returns:
            Response with follow-up questions and other information
        """
        manager = DialogueService._get_manager(db)
        return manager.process_message(user_id, message, language)
    
    @staticmethod
    def get_diagnosis(db: Session, user_id: str) -> Dict[str, Any]:
        """
        Get a diagnosis based on the current conversation state
        
        Args:
            db: Database session
            user_id: User identifier
            
        Returns:
            Diagnostic results
        """
        manager = DialogueService._get_manager(db)
        return manager.get_diagnosis_from_session(user_id)
    
    @staticmethod
    def end_conversation(db: Session, user_id: str) -> None:
        """
        End a diagnostic conversation
        
        Args:
            db: Database session
            user_id: User identifier
        """
        manager = DialogueService._get_manager(db)
        manager.close_session(user_id)