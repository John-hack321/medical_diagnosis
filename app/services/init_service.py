import logging
from sqlalchemy.orm import Session
from app.services.data_service import DataService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database(db: Session) -> None:
    """
    Initialize the database with data from CSV and other sources
    
    Args:
        db: Database session
    """
    logger.info("Initializing database with data")
    
    try:
        # Import data from CSV
        csv_path = "/home/john/Desktop/medical_diagnosis_app/app/data/datasets/Diseases_Symptoms.csv"
        DataService.import_csv_data(db, csv_path)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise