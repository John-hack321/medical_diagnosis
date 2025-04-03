import csv
import logging
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataService:
    @staticmethod
    def import_csv_data(db: Session, csv_path: str) -> None:
        """
        Import disease and symptom data from CSV file to database
        
        Args:
            db: Database session
            csv_path: Path to the CSV file
        """
        logger.info(f"Importing data from {csv_path}")
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Check if disease already exists
                    disease_name = row['Name']
                    existing_disease = db.query(Disease).filter_by(name=disease_name).first()
                    
                    if existing_disease:
                        disease = existing_disease
                        logger.info(f"Updating existing disease: {disease_name}")
                    else:
                        disease = Disease(name=disease_name)
                        logger.info(f"Creating new disease: {disease_name}")
                    
                    # Update disease information
                    disease.description = row.get('Description', '')
                    disease.treatment = row.get('Treatments', '')
                    disease.source_url = 'CSV Import'
                    
                    # Extract symptoms from the 'Symptoms' column
                    symptom_text = row.get('Symptoms', '')
                    symptom_list = [s.strip() for s in symptom_text.split(',')]
                    
                    for symptom_name in symptom_list:
                        if not symptom_name:
                            continue
                            
                        # Check if symptom already exists
                        symptom = db.query(Symptom).filter_by(name=symptom_name).first()
                        
                        if not symptom:
                            # Create new symptom
                            symptom = Symptom(name=symptom_name)
                            db.add(symptom)
                            db.flush()  # Generate ID without committing
                        
                        # Add to disease's symptoms if not already there
                        if symptom not in disease.symptoms:
                            disease.symptoms.append(symptom)
                    
                    # Add disease to session if it's new
                    if not existing_disease:
                        db.add(disease)
                
                # Commit all changes
                db.commit()
                logger.info("Successfully imported CSV data to database")
                
        except Exception as e:
            db.rollback()
            logger.error(f"Error importing CSV data: {str(e)}")
            raise

    @staticmethod
    def query_symptoms(db: Session, symptoms_list: List[str], limit: int = 5) -> Dict[str, Any]:
        """
        Query the database for diseases matching a list of symptoms
        
        Args:
            db: Database session
            symptoms_list: List of symptom names to query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary of possible diseases with confidence scores
        """
        # Find diseases that match these symptoms
        matching_diseases = {}
        
        for symptom_name in symptoms_list:
            # Find the symptom
            symptom = db.query(Symptom).filter(Symptom.name.ilike(f"%{symptom_name}%")).first()
            
            if symptom:
                # Find all diseases with this symptom
                for disease in symptom.diseases:
                    if disease.name not in matching_diseases:
                        matching_diseases[disease.name] = {
                            'matched_symptoms': 1,
                            'total_symptoms': len(disease.symptoms),
                            'disease_info': {
                                'description': disease.description,
                                'treatment': disease.treatment,
                                'prevention': disease.prevention if disease.prevention else ""
                            }
                        }
                    else:
                        matching_diseases[disease.name]['matched_symptoms'] += 1
        
        # Calculate confidence scores
        results = {}
        for disease_name, data in matching_diseases.items():
            # Confidence: ratio of matched symptoms to total symptoms
            confidence = data['matched_symptoms'] / max(1, data['total_symptoms'])
            
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
        
        return sorted_results