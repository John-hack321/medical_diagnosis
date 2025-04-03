from fastapi import FastAPI
from app.routers import auth, diagnosis
from app.models import disease, user
from app.database import engine

# Create database tables
disease.Base.metadata.create_all(bind=engine)
user.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Medical Diagnosis API",
    description="An API for diagnosing medical conditions based on symptoms",
    version="1.0.0"
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(diagnosis.router, prefix="/api", tags=["Diagnosis"])

@app.get("/")
def home():
    return {"message": "Welcome to the Medical Diagnostics API"}