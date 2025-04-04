from fastapi import FastAPI
from app.routers import auth, diagnosis, conversation
from app.models import disease, user
from app.database import engine
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/john/Desktop/medical_diagnosis_app/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create database tables
logger.info("Creating database tables if they don't exist")
disease.Base.metadata.create_all(bind=engine)
user.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Medical Diagnosis API",
    description="An API for diagnosing medical conditions based on symptoms using AI and NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Include routers
logger.info("Setting up API routes")
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(diagnosis.router, prefix="/api", tags=["Diagnosis"])
app.include_router(conversation.router, prefix="/conversation", tags=["Conversation"])

# Add middleware for CORS if needed
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/", tags=["Home"])
def home():
    """
    Root endpoint that provides basic information about the API
    """
    return {
        "message": "Welcome to the Medical Diagnostics API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "authentication": "/auth/login",
            "diagnosis": "/api/diagnose",
            "natural_language": "/api/diagnose-text",
            "conversation": "/conversation/message"
        },
        "description": "This API provides medical diagnostic capabilities using AI and natural language processing."
    }

@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint for monitoring
    """
    return {"status": "healthy"}