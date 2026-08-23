from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from agents.supervisor import Supervisor

app = FastAPI(
    title="Codebase Copilot API",
    description="API for the multi-agent codebase copilot",
    version="0.1.0"
)

try:
    supervisor = Supervisor()
    print("✅ Supervisor successfully initialized in FastAPI backend.")
except Exception as e:
    print(f"❌ Failed to initialize Supervisor: {e}")
    supervisor = None

class HealthResponse(BaseModel):
    status: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str = "unknown"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not supervisor:
        raise HTTPException(status_code=500, detail="Supervisor is not initialized.")
    
    try:
        initial_state = {"input_text": request.message, "intent": "", "final_output": ""}
        result_state = supervisor.graph.invoke(initial_state)
        
        return ChatResponse(
            response=result_state.get("final_output", "No response generated."),
            intent=result_state.get("intent", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend
app.mount("/", StaticFiles(directory="public", html=True), name="public")
