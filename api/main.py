from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.supervisor import Supervisor

app = FastAPI(
    title="Codebase Copilot API",
    description="API for the multi-agent codebase copilot",
    version="0.2.0",
)

try:
    supervisor = Supervisor()
    print("Supervisor initialised.")
except Exception as e:
    print(f"Failed to initialise Supervisor: {e}")
    supervisor = None


class HealthResponse(BaseModel):
    status: str
    supervisor_ready: bool
    indexed_chunks: int


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: str = "unknown"


@app.get("/health", response_model=HealthResponse)
def health_check():
    chunks = len(supervisor.searcher.chunks) if supervisor else 0
    return {"status": "ok", "supervisor_ready": supervisor is not None, "indexed_chunks": chunks}


# A plain `def` endpoint: FastAPI runs it in a worker thread. The graph call
# blocks for seconds (model inference + LLM HTTP calls); inside `async def`
# it would freeze the event loop and every other request with it.
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not supervisor:
        raise HTTPException(status_code=503, detail="Supervisor is not initialised. Check the server logs.")
    try:
        state = supervisor.run(request.message)
        return ChatResponse(response=state.get("final_output") or "No response generated.",
                            intent=state.get("intent") or "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mounted last so /api/* and /health match first
app.mount("/", StaticFiles(directory="public", html=True), name="public")
