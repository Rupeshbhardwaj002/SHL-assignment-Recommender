from fastapi import FastAPI, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.agent.controller import orchestrate_chat
from app.llm.generator import generate_final_response
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
    the next agent action and grounded recommendations.
    """
    try:
        print(f"\n--- NEW CHAT REQUEST ({len(request.messages)} turns) ---")
        
        # 1. Orchestrate: Extract intent, constraints, and retrieve catalog docs
        orchestration = orchestrate_chat(request.messages)
        
        # FAST PATH: If the controller triggered a hardcoded safety REFUSE or a CLARIFY rule, 
        # it will bypass the LLM generator entirely to save time.
        if "reply" in orchestration:
            print("Action: Executing Fast-Path Rule (Refuse/Clarify)")
            return ChatResponse(
                reply=orchestration["reply"],
                recommendations=orchestration["recommendations"],
                end_of_conversation=orchestration["end_of_conversation"]
            )
            
        # 2. Generate: If intent is RECOMMEND, REFINE, or COMPARE, use the LLM
        print("Action: Generating Grounded LLM Response")
        final_json = generate_final_response(
            intent=orchestration["intent"],
            state=orchestration["state"],
            raw_docs=orchestration["raw_docs"],
            messages=request.messages
        )
        
        print("Response Generated Successfully.")
        return ChatResponse(**final_json)
        
    except Exception as e:
        print(f"API Error: {e}")
        # Always return valid schema even on crash
        return ChatResponse(
            reply="I am experiencing technical difficulties. Please try again.",
            recommendations=[],
            end_of_conversation=False
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)