from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(request: LoginRequest):
    return {"message": f"Logged in as {request.username}"}
