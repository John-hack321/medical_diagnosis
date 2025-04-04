# Medical Diagnosis Assistant

An AI-powered medical diagnostics system designed for use in Level 2 and Level 3 hospitals in Kenya. This system uses multiple AI approaches including natural language processing, decision trees, neural networks, and Bayesian reasoning to provide diagnostic suggestions based on reported symptoms.

## Key Features

- **Multi-approach AI Diagnosis**: Combines rule-based, machine learning, and Bayesian approaches
- **Natural Language Processing**: Extract symptoms from natural language text in English and Swahili
- **Conversational Interface**: Interactive dialogue system for symptom collection and follow-up
- **Medical Ontology**: Standardized medical terminology for accurate symptom matching
- **Web Scraping**: Automated collection of medical knowledge from authoritative sources
- **Safety Features**: Emergency detection and uncertainty quantification

## System Architecture

The system is organized into the following components:

- **AI Service**: Core diagnostic engine that combines multiple approaches
- **Dialogue Service**: Manages conversations and follow-up questions
- **Ontology Service**: Standardizes medical terminology and relationships
- **Bayesian Service**: Probabilistic reasoning for diagnosis under uncertainty
- **Scraper Service**: Collects medical knowledge from health websites

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL
- **AI/ML**: PyTorch, Scikit-learn
- **NLP**: Transformers (Hugging Face)
- **Web Scraping**: Requests, BeautifulSoup4

## API Endpoints

- `/api/diagnose` - Diagnose based on a list of symptoms
- `/api/diagnose-text` - Extract symptoms from text and diagnose
- `/api/treatment` - Get treatment information for a disease
- `/api/treatment-insights` - Get AI-enhanced insights about treatments
- `/conversation/message` - Process a message in a diagnostic conversation
- `/conversation/diagnose` - Generate diagnosis from conversation

## Installation and Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure PostgreSQL database connection in `app/database.py`
4. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

## Data Sources

The system collects medical data from various sources:

- Kenya Ministry of Health
- WHO Africa
- Nigeria Federal Ministry of Health
- South Africa Department of Health
- Ghana Health Service
- Africa CDC
- African medical journals

## Models

Multiple AI models are used for diagnosis:

- Neural Network: Deep learning for pattern recognition
- Decision Tree: Interpretable rules for diagnosis
- Bayesian Network: Probabilistic reasoning for uncertainty

## Usage Examples

**Symptom-based diagnosis:**

```bash
curl -X POST "http://localhost:8000/api/diagnose" -H "Content-Type: application/json" -d '{"symptoms": ["fever", "cough", "headache"]}'
```

**Natural language diagnosis:**

```bash
curl -X POST "http://localhost:8000/api/diagnose-text" -H "Content-Type: application/json" -d '{"text": "I have been having a fever and headache for the past two days. Also feeling very tired.", "language": "english"}'
```

**Conversational diagnosis:**

```bash
curl -X POST "http://localhost:8000/conversation/message" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "I have been having a fever for 3 days", "language": "english"}'
```

## Safety and Ethics

- The system includes comprehensive disclaimers
- Emergency symptoms trigger appropriate warnings
- Uncertainty is explicitly quantified and communicated
- The system is designed as a decision support tool, not a replacement for medical professionals

## License

This project is designed for public health benefit and is available under [appropriate license].

## Acknowledgments

- University of Nairobi
- Faculty of Science and Technology
- Department of Computing and Informatics

Kenya Ministry of Health - WHO Africa - Nigeria Federal Ministry of Health - South Africa Department of Health - Ghana Health Service - Africa CDC - African medical journals

     ## Models

     Multiple AI models are used for diagnosis:

     - Neural Network: Deep learning for pattern recognition
     - Decision Tree: Interpretable rules for diagnosis
     - Bayesian Network: Probabilistic reasoning for uncertainty

     ## Usage Examples

     **Symptom-based diagnosis:**
     ```bash
     curl -X POST "http://localhost:8000/api/diagnose" -H "Content-Type: application/json" -d '{"symptoms": ["fever", "cough", "headache"]}'
     ```

     **Natural language diagnosis:**
     ```bash
     curl -X POST "http://localhost:8000/api/diagnose-text" -H "Content-Type: application/json" -d '{"text": "I have been having a fever and headache for the past two days. Also feeling very tired.", "language": "english"}'
     ```

     **Conversational diagnosis:**
     ```bash
     curl -X POST "http://localhost:8000/conversation/message" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "I have been having a fever for 3 days", "language": "english"}'
     ```

     ## Safety and Ethics

     - The system includes comprehensive disclaimers
     - Emergency symptoms trigger appropriate warnings
     - Uncertainty is explicitly quantified and communicated
     - The system is designed as a decision support tool, not a replacement for medical professionals

     ## License

     This project is designed for public health benefit and is available under [appropriate license].

     ## Acknowledgments

     - University of Nairobi
     - Faculty of Science and Technology
     - Department of Computing and Informatics
