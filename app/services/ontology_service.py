import logging
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalOntology:
    """Medical ontology for standardizing medical terms and relationships"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Initialize data structures for the ontology
        self.symptom_map = {}  # Maps symptom variants to canonical forms
        self.disease_map = {}  # Maps disease variants to canonical forms
        self.symptom_categories = {}  # Categorizes symptoms by body system
        self.disease_categories = {}  # Categorizes diseases by medical specialty
        self.symptom_relations = {}  # Relations between symptoms (e.g., "headache" is related to "migraine")
        self.disease_relations = {}  # Relations between diseases (e.g., "type 2 diabetes" is related to "insulin resistance")
        self.icd10_mapping = {}  # Maps diseases to ICD-10 codes
        
        # Initialize the ontology
        self._initialize_ontology()
    
    def _initialize_ontology(self) -> None:
        """Initialize the medical ontology with data from the database"""
        try:
            logger.info("Initializing medical ontology")
            
            # Load symptoms and build synonym mappings
            symptoms = self.db.query(Symptom).all()
            for symptom in symptoms:
                canonical = symptom.name.lower()
                self.symptom_map[canonical] = canonical
                
                # Generate common variants and synonyms
                variants = self._generate_symptom_variants(canonical)
                for variant in variants:
                    self.symptom_map[variant] = canonical
            
            # Load diseases and build synonym mappings
            diseases = self.db.query(Disease).all()
            for disease in diseases:
                canonical = disease.name.lower()
                self.disease_map[canonical] = canonical
                
                # Generate common variants and synonyms
                variants = self._generate_disease_variants(canonical)
                for variant in variants:
                    self.disease_map[variant] = canonical
            
            # Categorize symptoms by body system (simplified example)
            self._categorize_symptoms()
            
            # Categorize diseases by medical specialty (simplified example)
            self._categorize_diseases()
            
            # Build symptom relations
            self._build_symptom_relations()
            
            # Build disease relations
            self._build_disease_relations()
            
            # Map diseases to ICD-10 codes (simplified example)
            self._build_icd10_mapping()
            
            logger.info(f"Medical ontology initialized with {len(self.symptom_map)} symptom terms and {len(self.disease_map)} disease terms")
            
        except Exception as e:
            logger.error(f"Error initializing medical ontology: {str(e)}")
            raise
    
    def _generate_symptom_variants(self, symptom: str) -> List[str]:
        """Generate common variants and synonyms for a symptom"""
        variants = []
        
        # Add plural/singular forms
        if symptom.endswith('s') and not symptom.endswith('ss'):
            variants.append(symptom[:-1])  # Remove trailing 's'
        else:
            variants.append(symptom + 's')  # Add trailing 's'
        
        # Add common prefixes/suffixes
        if not symptom.startswith('severe '):
            variants.append('severe ' + symptom)
        if not symptom.startswith('mild '):
            variants.append('mild ' + symptom)
        if not symptom.endswith(' pain'):
            if 'ache' in symptom:
                variants.append(symptom.replace('ache', 'pain'))
        
        # Add specific synonyms for common symptoms
        if symptom == 'headache':
            variants.extend(['head pain', 'cephalgia'])
        elif symptom == 'fever':
            variants.extend(['high temperature', 'pyrexia', 'febrile'])
        elif symptom == 'cough':
            variants.extend(['coughing', 'hacking'])
        elif symptom == 'diarrhea':
            variants.extend(['loose stool', 'watery stool', 'loose bowel movement'])
        elif symptom == 'fatigue':
            variants.extend(['tiredness', 'exhaustion', 'lethargy', 'malaise'])
        
        return variants
    
    def _generate_disease_variants(self, disease: str) -> List[str]:
        """Generate common variants and synonyms for a disease"""
        variants = []
        
        # Add common prefixes/suffixes
        if not disease.startswith('chronic '):
            variants.append('chronic ' + disease)
        if not disease.startswith('acute '):
            variants.append('acute ' + disease)
        
        # Add specific synonyms for common diseases
        if disease == 'malaria':
            variants.extend(['malarial infection', 'plasmodium infection'])
        elif disease == 'tuberculosis':
            variants.extend(['tb', 'koch disease', 'consumption'])
        elif disease == 'hypertension':
            variants.extend(['high blood pressure', 'htn'])
        elif disease == 'diabetes':
            variants.extend(['diabetes mellitus', 'high blood sugar', 'hyperglycemia'])
        elif disease == 'influenza':
            variants.extend(['flu', 'grippe'])
        
        return variants
    
    def _categorize_symptoms(self) -> None:
        """Categorize symptoms by body system"""
        # This is a simplified implementation
        # In a real system, this would be more comprehensive and loaded from a database
        
        body_systems = {
            'respiratory': ['cough', 'shortness of breath', 'chest congestion', 'wheezing', 'sore throat'],
            'digestive': ['nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal pain'],
            'nervous': ['headache', 'dizziness', 'seizure', 'confusion', 'numbness'],
            'cardiovascular': ['chest pain', 'palpitations', 'edema', 'hypertension'],
            'musculoskeletal': ['joint pain', 'muscle pain', 'back pain', 'stiffness'],
            'dermatological': ['rash', 'itching', 'hives', 'discoloration'],
            'general': ['fever', 'fatigue', 'weight loss', 'night sweats', 'malaise']
        }
        
        # Map each symptom to its body system
        for system, symptoms in body_systems.items():
            for symptom in symptoms:
                if symptom in self.symptom_map:
                    canonical = self.symptom_map[symptom]
                    self.symptom_categories[canonical] = system
    
    def _categorize_diseases(self) -> None:
        """Categorize diseases by medical specialty"""
        # This is a simplified implementation
        # In a real system, this would be more comprehensive and loaded from a database
        
        specialties = {
            'infectious_disease': ['malaria', 'tuberculosis', 'hiv/aids', 'pneumonia', 'typhoid', 'cholera'],
            'cardiology': ['hypertension', 'coronary artery disease', 'heart failure', 'arrhythmia'],
            'endocrinology': ['diabetes', 'hypothyroidism', 'hyperthyroidism'],
            'gastroenterology': ['gastritis', 'peptic ulcer disease', 'diarrheal diseases'],
            'pulmonology': ['asthma', 'chronic obstructive pulmonary disease', 'respiratory infections'],
            'neurology': ['migraine', 'epilepsy', 'parkinson disease', 'meningitis'],
            'oncology': ['cancer', 'leukemia', 'lymphoma']
        }
        
        # Map each disease to its medical specialty
        for specialty, diseases in specialties.items():
            for disease in diseases:
                if disease in self.disease_map:
                    canonical = self.disease_map[disease]
                    self.disease_categories[canonical] = specialty
    
    def _build_symptom_relations(self) -> None:
        """Build relations between symptoms"""
        # This is a simplified implementation
        # In a real system, this would be more comprehensive and loaded from a database
        
        # Define some symptom groups and relations
        symptom_relations = {
            'headache': ['dizziness', 'vision changes', 'nausea'],
            'fever': ['chills', 'sweating', 'fatigue'],
            'cough': ['sore throat', 'chest congestion', 'shortness of breath'],
            'abdominal pain': ['nausea', 'vomiting', 'diarrhea', 'constipation'],
            'joint pain': ['muscle pain', 'stiffness', 'swelling']
        }
        
        # Build the relations
        for symptom, related in symptom_relations.items():
            if symptom in self.symptom_map:
                canonical = self.symptom_map[symptom]
                related_canonical = []
                
                for rel in related:
                    if rel in self.symptom_map:
                        related_canonical.append(self.symptom_map[rel])
                
                self.symptom_relations[canonical] = related_canonical
    
    def _build_disease_relations(self) -> None:
        """Build relations between diseases"""
        # This is a simplified implementation
        # In a real system, this would be more comprehensive and loaded from a database
        
        # Define some disease relations
        disease_relations = {
            'diabetes': ['hypertension', 'obesity', 'heart disease'],
            'tuberculosis': ['hiv/aids', 'pneumonia'],
            'malaria': ['anemia', 'jaundice'],
            'hypertension': ['heart disease', 'stroke', 'kidney disease']
        }
        
        # Build the relations
        for disease, related in disease_relations.items():
            if disease in self.disease_map:
                canonical = self.disease_map[disease]
                related_canonical = []
                
                for rel in related:
                    if rel in self.disease_map:
                        related_canonical.append(self.disease_map[rel])
                
                self.disease_relations[canonical] = related_canonical
    
    def _build_icd10_mapping(self) -> None:
        """Map diseases to ICD-10 codes"""
        # This is a simplified implementation
        # In a real system, this would be more comprehensive and loaded from a database
        
        # Define some ICD-10 codes for common diseases
        icd10_codes = {
            'malaria': 'B54',
            'tuberculosis': 'A15',
            'hiv/aids': 'B20',
            'pneumonia': 'J18',
            'typhoid': 'A01.0',
            'cholera': 'A00',
            'diabetes': 'E11',  # Type 2 diabetes
            'hypertension': 'I10',
            'cancer': 'C80',  # Malignant neoplasm, unspecified
            'diarrheal diseases': 'A09'
        }
        
        # Build the mapping
        for disease, code in icd10_codes.items():
            if disease in self.disease_map:
                canonical = self.disease_map[disease]
                self.icd10_mapping[canonical] = code
    
    def standardize_symptom(self, symptom: str) -> Optional[str]:
        """
        Convert a symptom term to its canonical form
        
        Args:
            symptom: A symptom term that may be a variant or synonym
            
        Returns:
            The canonical form of the symptom, or None if not recognized
        """
        symptom_lower = symptom.lower()
        
        # Direct match
        if symptom_lower in self.symptom_map:
            return self.symptom_map[symptom_lower]
        
        # Try to find a partial match
        for known_symptom in self.symptom_map:
            if known_symptom in symptom_lower or symptom_lower in known_symptom:
                return self.symptom_map[known_symptom]
        
        # No match found
        return None
    
    def standardize_disease(self, disease: str) -> Optional[str]:
        """
        Convert a disease term to its canonical form
        
        Args:
            disease: A disease term that may be a variant or synonym
            
        Returns:
            The canonical form of the disease, or None if not recognized
        """
        disease_lower = disease.lower()
        
        # Direct match
        if disease_lower in self.disease_map:
            return self.disease_map[disease_lower]
        
        # Try to find a partial match
        for known_disease in self.disease_map:
            if known_disease in disease_lower or disease_lower in known_disease:
                return self.disease_map[known_disease]
        
        # No match found
        return None
    
    def get_related_symptoms(self, symptom: str) -> List[str]:
        """
        Get symptoms related to the given symptom
        
        Args:
            symptom: A symptom term
            
        Returns:
            List of related symptoms
        """
        canonical = self.standardize_symptom(symptom)
        if not canonical:
            return []
        
        if canonical in self.symptom_relations:
            return self.symptom_relations[canonical]
        
        return []
    
    def get_related_diseases(self, disease: str) -> List[str]:
        """
        Get diseases related to the given disease
        
        Args:
            disease: A disease term
            
        Returns:
            List of related diseases
        """
        canonical = self.standardize_disease(disease)
        if not canonical:
            return []
        
        if canonical in self.disease_relations:
            return self.disease_relations[canonical]
        
        return []
    
    def get_symptom_category(self, symptom: str) -> Optional[str]:
        """
        Get the body system category for a symptom
        
        Args:
            symptom: A symptom term
            
        Returns:
            The body system category, or None if not categorized
        """
        canonical = self.standardize_symptom(symptom)
        if not canonical:
            return None
        
        if canonical in self.symptom_categories:
            return self.symptom_categories[canonical]
        
        return None
    
    def get_disease_category(self, disease: str) -> Optional[str]:
        """
        Get the medical specialty category for a disease
        
        Args:
            disease: A disease term
            
        Returns:
            The medical specialty category, or None if not categorized
        """
        canonical = self.standardize_disease(disease)
        if not canonical:
            return None
        
        if canonical in self.disease_categories:
            return self.disease_categories[canonical]
        
        return None
    
    def get_icd10_code(self, disease: str) -> Optional[str]:
        """
        Get the ICD-10 code for a disease
        
        Args:
            disease: A disease term
            
        Returns:
            The ICD-10 code, or None if not mapped
        """
        canonical = self.standardize_disease(disease)
        if not canonical:
            return None
        
        if canonical in self.icd10_mapping:
            return self.icd10_mapping[canonical]
        
        return None
    
    def standardize_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Convert a list of symptom terms to their canonical forms
        
        Args:
            symptoms: List of symptom terms
            
        Returns:
            List of canonical symptom terms (excluding any that were not recognized)
        """
        standardized = []
        for symptom in symptoms:
            canonical = self.standardize_symptom(symptom)
            if canonical:
                standardized.append(canonical)
        
        return standardized


class OntologyService:
    """Service interface for medical ontology functionality"""
    
    # Cache the ontology instance
    _ontology_instance = None
    
    @classmethod
    def _get_ontology(cls, db: Session) -> MedicalOntology:
        """Get or create a MedicalOntology instance"""
        if cls._ontology_instance is None:
            cls._ontology_instance = MedicalOntology(db)
        return cls._ontology_instance
    
    @staticmethod
    def standardize_symptoms(db: Session, symptoms: List[str]) -> List[str]:
        """
        Standardize a list of symptoms to their canonical forms
        
        Args:
            db: Database session
            symptoms: List of symptom terms
            
        Returns:
            List of standardized symptom terms
        """
        ontology = OntologyService._get_ontology(db)
        return ontology.standardize_symptoms(symptoms)
    
    @staticmethod
    def get_related_symptoms(db: Session, symptom: str) -> List[str]:
        """
        Get symptoms related to the given symptom
        
        Args:
            db: Database session
            symptom: A symptom term
            
        Returns:
            List of related symptoms
        """
        ontology = OntologyService._get_ontology(db)
        return ontology.get_related_symptoms(symptom)
    
    @staticmethod
    def get_icd10_for_disease(db: Session, disease: str) -> Optional[str]:
        """
        Get the ICD-10 code for a disease
        
        Args:
            db: Database session
            disease: A disease term
            
        Returns:
            The ICD-10 code, or None if not mapped
        """
        ontology = OntologyService._get_ontology(db)
        return ontology.get_icd10_code(disease)
    
    @staticmethod
    def categorize_symptoms(db: Session, symptoms: List[str]) -> Dict[str, List[str]]:
        """
        Categorize symptoms by body system
        
        Args:
            db: Database session
            symptoms: List of symptom terms
            
        Returns:
            Dictionary mapping body systems to lists of symptoms
        """
        ontology = OntologyService._get_ontology(db)
        
        categorized = {}
        for symptom in symptoms:
            canonical = ontology.standardize_symptom(symptom)
            if not canonical:
                continue
            
            category = ontology.get_symptom_category(canonical) or 'uncategorized'
            
            if category not in categorized:
                categorized[category] = []
            
            categorized[category].append(canonical)
        
        return categorized