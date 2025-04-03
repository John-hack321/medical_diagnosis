from fastapi import FastAPI
from fastapi import FastAPI
from app.routers import auth

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Medical Diagnostics API"}





app = FastAPI()

# Include auth routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

@app.get("/")
def home():
    return {"message": "Welcome to the Medical Diagnostics API"}
