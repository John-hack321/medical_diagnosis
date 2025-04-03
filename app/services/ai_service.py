import logging
from typing import List, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIService:
    @staticmethod
    def diagnose(db: Session, symptoms: List[str], limit: int = 5) -> Dict[str, Any]:
        """
        Provide diagnosis based on symptoms using a simple matching algorithm
        
        Args:
            db: Database session
            symptoms: List of symptom names reported by user
            limit: Maximum number of results to return
            
        Returns:
            Dictionary of possible diagnoses with confidence scores
        """
        logger.info(f"Generating diagnosis for symptoms: {symptoms}")
        
        # Find all symptoms in database that match
        matching_diseases = {}
        matched_symptoms = {}
        
        for symptom_name in symptoms:
            # Use partial matching to find symptoms
            db_symptoms = db.query(Symptom).filter(
                Symptom.name.ilike(f"%{symptom_name}%")
            ).all()
            
            for symptom in db_symptoms:
                matched_symptoms[symptom_name] = symptom.name
                
                # Find all diseases with this symptom
                for disease in symptom.diseases:
                    if disease.name not in matching_diseases:
                        matching_diseases[disease.name] = {
                            'matched_symptoms': [symptom.name],
                            'total_symptoms': len(disease.symptoms),
                            'disease_info': {
                                'id': disease.id,
                                'description': disease.description,
                                'treatment': disease.treatment,
                                'prevention': disease.prevention if disease.prevention else ""
                            }
                        }
                    else:
                        if symptom.name not in matching_diseases[disease.name]['matched_symptoms']:
                            matching_diseases[disease.name]['matched_symptoms'].append(symptom.name)
        
        # Calculate confidence scores
        results = {}
        for disease_name, data in matching_diseases.items():
            # Basic confidence formula: number of matched symptoms / total symptoms
            # With a weighting factor to prioritize diseases with more matches 
            matched_count = len(data['matched_symptoms'])
            total_count = max(1, data['total_symptoms'])
            
            # Adjust confidence based on symptoms matched and total symptoms
            # This formula balances between complete match and partial match
            match_ratio = matched_count / total_count
            symptom_coverage = matched_count / max(1, len(symptoms))
            
            # Combined score with weights (can be adjusted)
            confidence = (0.7 * match_ratio) + (0.3 * symptom_coverage)
            
            results[disease_name] = {
                'confidence': round(confidence * 100, 2),  # Convert to percentage
                'matched_symptoms': data['matched_symptoms'],
                'total_symptoms': data['total_symptoms'],
                'info': data['disease_info']
            }
        
        # Sort by confidence and limit results
        sorted_results = dict(sorted(
            results.items(), 
            key=lambda item: item[1]['confidence'], 
            reverse=True
        )[:limit])
        
        # Add input symptoms with their matches
        return {
            'diagnoses': sorted_results,
            'symptom_matches': matched_symptoms
        }
    
    @staticmethod
    def get_treatment(db: Session, disease_id: int) -> Dict[str, Any]:
        """
        Get treatment information for a specific disease
        
        Args:
            db: Database session
            disease_id: ID of the disease
            
        Returns:
            Treatment information
        """
        disease = db.query(Disease).filter_by(id=disease_id).first()
        
        if not disease:
            return {'error': 'Disease not found'}
        
        return {
            'name': disease.name,
            'treatment': disease.treatment,
            'prevention': disease.prevention if disease.prevention else "",
            'risk_factors': disease.risk_factors if disease.risk_factors else ""
        }