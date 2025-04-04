import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BayesianDiagnosticEngine:
    """Bayesian network for medical diagnosis"""
    
    def __init__(self, db: Session):
        self.db = db
        self.diseases = {}
        self.symptoms = set()
        self.disease_prior = {}
        self.symptom_given_disease = {}
        self.symptom_given_not_disease = {}
        
        # Initialize the network
        self._initialize_network()
    
    def _initialize_network(self) -> None:
        """Initialize the Bayesian network with data from the database"""
        try:
            logger.info("Initializing Bayesian diagnostic network")
            
            # Get all diseases and their symptoms
            all_diseases = self.db.query(Disease).all()
            
            # Build the network structure
            disease_count = 0
            symptom_count = 0
            
            for disease in all_diseases:
                if disease.symptoms:
                    disease_count += 1
                    disease_name = disease.name.lower()
                    self.diseases[disease_name] = {
                        'id': disease.id,
                        'symptoms': [symptom.name.lower() for symptom in disease.symptoms]
                    }
                    
                    # Add to our symptom set
                    for symptom in disease.symptoms:
                        self.symptoms.add(symptom.name.lower())
                        symptom_count += 1
            
            # Calculate disease priors (simple approach: assume uniform distribution)
            # In a real system, these would be based on prevalence data
            for disease_name in self.diseases:
                self.disease_prior[disease_name] = 1.0 / len(self.diseases)
            
            # Calculate conditional probabilities P(symptom|disease) and P(symptom|not disease)
            for symptom in self.symptoms:
                self.symptom_given_disease[symptom] = {}
                self.symptom_given_not_disease[symptom] = {}
                
                for disease_name in self.diseases:
                    # If symptom is associated with disease, high probability (0.7-0.9)
                    # If not, low probability (0.01-0.1)
                    # These would ideally be learned from data
                    if symptom in self.diseases[disease_name]['symptoms']:
                        # P(symptom|disease) - value between 0.7 and 0.9 based on how specific it is
                        # For simplicity, just using 0.8
                        self.symptom_given_disease[symptom][disease_name] = 0.8
                        
                        # P(symptom|not disease) - estimated based on symptom frequency
                        # For simplicity, using 0.1
                        self.symptom_given_not_disease[symptom][disease_name] = 0.1
                    else:
                        # Symptom not associated with disease
                        self.symptom_given_disease[symptom][disease_name] = 0.05
                        self.symptom_given_not_disease[symptom][disease_name] = 0.2
            
            logger.info(f"Bayesian network initialized with {disease_count} diseases and {len(self.symptoms)} unique symptoms")
            
        except Exception as e:
            logger.error(f"Error initializing Bayesian network: {str(e)}")
            raise
    
    def diagnose(self, symptoms: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Generate diagnosis using Bayesian inference
        
        Args:
            symptoms: List of symptoms reported by the patient
            top_n: Number of top diagnoses to return
            
        Returns:
            List of potential diagnoses with probabilities
        """
        try:
            # Normalize input symptoms
            normalized_symptoms = [s.lower() for s in symptoms]
            
            # Calculate posterior probability for each disease
            disease_probabilities = {}
            
            for disease_name in self.diseases:
                # Start with prior probability
                prior = self.disease_prior[disease_name]
                
                # Calculate likelihood
                likelihood_disease = 1.0
                likelihood_not_disease = 1.0
                
                for symptom in self.symptoms:
                    if symptom in normalized_symptoms:
                        # Symptom is present
                        if disease_name in self.symptom_given_disease[symptom]:
                            likelihood_disease *= self.symptom_given_disease[symptom][disease_name]
                            likelihood_not_disease *= self.symptom_given_not_disease[symptom][disease_name]
                    else:
                        # Symptom is absent - inverse probability
                        if disease_name in self.symptom_given_disease[symptom]:
                            likelihood_disease *= (1 - self.symptom_given_disease[symptom][disease_name])
                            likelihood_not_disease *= (1 - self.symptom_given_not_disease[symptom][disease_name])
                
                # Calculate posterior using Bayes' rule: P(disease|symptoms) = P(symptoms|disease) * P(disease) / P(symptoms)
                # We can skip the denominator for comparison purposes
                posterior = likelihood_disease * prior
                
                # Store result
                disease_probabilities[disease_name] = {
                    'probability': posterior,
                    'disease_id': self.diseases[disease_name]['id'],
                    'matched_symptoms': [s for s in normalized_symptoms if s in self.diseases[disease_name]['symptoms']]
                }
            
            # Normalize probabilities
            total_probability = sum(item['probability'] for item in disease_probabilities.values())
            if total_probability > 0:
                for disease_name in disease_probabilities:
                    disease_probabilities[disease_name]['probability'] /= total_probability
                    # Convert to percentage
                    disease_probabilities[disease_name]['probability'] *= 100
            
            # Sort by probability and get top N
            sorted_diagnoses = sorted(
                disease_probabilities.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:top_n]
            
            # Format results
            results = []
            for disease_name, data in sorted_diagnoses:
                # Only include diseases with non-zero probability
                if data['probability'] > 0.5:  # Threshold to filter out very unlikely diseases
                    disease = self.db.query(Disease).filter_by(id=data['disease_id']).first()
                    results.append({
                        'disease': disease_name,
                        'probability': round(data['probability'], 2),
                        'matched_symptoms': data['matched_symptoms'],
                        'disease_info': {
                            'id': disease.id,
                            'description': disease.description,
                            'treatment': disease.treatment,
                            'prevention': disease.prevention if disease.prevention else ""
                        }
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Bayesian diagnosis: {str(e)}")
            return []