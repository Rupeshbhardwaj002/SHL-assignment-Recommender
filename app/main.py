# FastAPI application instance & server startup
from fastapi import FastAPI, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
import uvicorn

app = FastAPI(title="SHL Assessment Recommender")

@app.get("/health")
async def health_check():
    """Readiness endpoint required by the evaluator."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Stateless chat endpoint. Takes the full conversation history and returns
    the next agent action and potential recommendations.
    """
    try:
        # TODO: Pass request.messages to app/agent/controller.py
        
        # Placeholder response to verify the server works
        return ChatResponse(
            reply="I am the SHL agent. How can I help you find an assessment?",
            recommendations=[],
            end_of_conversation=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)