import requests
from bs4 import BeautifulSoup
import time
import re
import logging
from urllib.parse import urljoin
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.disease import Disease, Symptom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalKnowledgeScraper:
    def __init__(self, db: Session):
        """
        Initialize the scraper with database session
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.headers = {
            'User-Agent': 'Medical Knowledge Research Bot (academic research project)',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        self.base_delay = 5  # seconds between requests
        self.medical_terms = self._load_seed_medical_terms()
        
    def _load_seed_medical_terms(self) -> List[str]:
        """Load initial set of medical terms to search for"""
        # In a real implementation, this could be loaded from a file
        return [
            "malaria", "tuberculosis", "HIV/AIDS", "pneumonia", 
            "typhoid", "cholera", "diabetes", "hypertension",
            "cancer", "respiratory infections", "diarrheal diseases"
        ]
    
    def scrape_ministry_of_health(self) -> None:
        """Scrape Kenya Ministry of Health website for disease information"""
        base_url = "https://www.health.go.ke/"
        logger.info(f"Starting to scrape {base_url}")
        
        try:
            response = requests.get(base_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find and follow links to disease information pages
            for link in soup.find_all('a', href=True):
                for term in self.medical_terms:
                    if term.lower() in link.text.lower() or term.lower() in link['href'].lower():
                        disease_url = urljoin(base_url, link['href'])
                        self._extract_disease_info(disease_url, term)
                        time.sleep(self.base_delay)  # Respectful delay
        
        except Exception as e:
            logger.error(f"Error scraping Ministry of Health: {str(e)}")
    
    def scrape_who_africa(self) -> None:
        """Scrape WHO Africa for disease information"""
        base_url = "https://www.afro.who.int/health-topics"
        logger.info(f"Starting to scrape {base_url}")
        
        try:
            response = requests.get(base_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find disease links
            for link in soup.find_all('a', href=True):
                for term in self.medical_terms:
                    if term.lower() in link.text.lower():
                        disease_url = urljoin(base_url, link['href'])
                        self._extract_disease_info(disease_url, term)
                        time.sleep(self.base_delay)
                
        except Exception as e:
            logger.error(f"Error scraping WHO Africa: {str(e)}")
    
    def _extract_disease_info(self, url: str, disease_term: str) -> None:
        """Extract structured disease information and save to database"""
        try:
            logger.info(f"Extracting information for {disease_term} from {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if this disease already exists
            existing_disease = self.db.query(Disease).filter_by(name=disease_term).first()
            
            if existing_disease:
                disease = existing_disease
                logger.info(f"Updating existing disease: {disease_term}")
            else:
                disease = Disease(name=disease_term)
                logger.info(f"Creating new disease: {disease_term}")
            
            # Update disease information
            disease.source_url = url
            disease.description = self._extract_text_section(soup, ['description', 'about', 'overview'])
            disease.treatment = self._extract_text_section(soup, ['treatment', 'management'])
            disease.prevention = self._extract_text_section(soup, ['prevention', 'precaution'])
            disease.risk_factors = self._extract_text_section(soup, ['risk', 'factor'])
            disease.epidemiology = self._extract_text_section(soup, ['epidemiology', 'statistics', 'prevalence'])
            disease.scrape_date = time.strftime('%Y-%m-%d')
            
            # Extract symptoms and create relationships
            symptom_text = self._extract_text_section(soup, ['symptom', 'signs'])
            extracted_symptoms = self._extract_symptoms_from_text(symptom_text)
            
            for symptom_name in extracted_symptoms:
                # Check if this symptom already exists
                symptom = self.db.query(Symptom).filter_by(name=symptom_name).first()
                
                if not symptom:
                    # Create new symptom
                    symptom = Symptom(name=symptom_name)
                    self.db.add(symptom)
                
                # Add to disease's symptoms if not already there
                if symptom not in disease.symptoms:
                    disease.symptoms.append(symptom)
            
            # Add disease to the session if it's new
            if not existing_disease:
                self.db.add(disease)
            
            # Commit changes to database
            self.db.commit()
            logger.info(f"Successfully saved information for {disease_term}")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error extracting disease info from {url}: {str(e)}")
    
    def _extract_text_section(self, soup: BeautifulSoup, keywords: List[str]) -> str:
        """Extract text from sections that match any of the keywords"""
        relevant_text = []
        
        # Look for headings with keywords
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if any(keyword in heading.text.lower() for keyword in keywords):
                # Get the next sibling elements until the next heading
                current = heading.find_next_sibling()
                while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if current.name in ['p', 'li', 'div'] and current.text.strip():
                        relevant_text.append(current.text.strip())
                    current = current.find_next_sibling()
        
        # Also look for paragraphs with keywords
        for para in soup.find_all(['p', 'li']):
            if any(keyword in para.text.lower() for keyword in keywords):
                relevant_text.append(para.text.strip())
        
        return ' '.join(relevant_text)
    
    def _extract_symptoms_from_text(self, text: str) -> List[str]:
        """Extract symptoms from text using simple patterns"""
        symptoms = []
        
        # Look for common symptom patterns
        patterns = [
            r'symptoms include ([\w\s,]+)',
            r'signs and symptoms.{1,30}([\w\s,]+)',
            r'(fever|pain|cough|headache|fatigue|nausea|vomiting|diarrhea|ache)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # If the pattern has groups
                    for item in match:
                        symptoms.extend(self._split_symptom_text(item))
                else:
                    # If the pattern doesn't have groups
                    symptoms.extend(self._split_symptom_text(match))
        
        return list(set(symptoms))  # Remove duplicates
    
    def _split_symptom_text(self, text: str) -> List[str]:
        """Split a text containing multiple symptoms into individual symptoms"""
        # Split by common delimiters
        items = re.split(r',|\band\b|\bor\b|;', text)
        return [item.strip().lower() for item in items if item.strip()]
    
    def run_scraper(self) -> None:
        """Run the complete scraping process"""
        logger.info("Starting medical knowledge scraping process")
        try:
            self.scrape_ministry_of_health()
            self.scrape_who_africa()
            # Add more scraping methods as needed
            
            logger.info("Scraping process completed")
        except Exception as e:
            logger.error(f"Error in scraping process: {str(e)}")
            self.db.rollback()