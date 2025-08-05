from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
app = FastAPI(title="DocuMind AI - HackRX Edition")
class SimpleRequest(BaseModel):
    documents: List[str]
    questions: List[str]
@app.get("/")
def root():
    return {"message": "DocuMind AI Backend - Ready for HackRX", "status": "active"}
@app.post("/api/v1/hackrx/run")
def hackrx_endpoint(request: SimpleRequest):
    # Simple response for hackathon testing
    answers = []
    for question in request.questions:
        answers.append(f"Sample answer for: {question}")
    return {"answers": answers}
@app.get("/health")
def health():
    return {"status": "healthy"}
