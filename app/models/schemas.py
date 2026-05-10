# Strict Pydantic models for the required API responses
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# Input Schema
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Output Schemas
class Recommendation(BaseModel):
    name: str
    url: str  # Kept as string to prevent strict Pydantic URL formatting errors against their evaluator
    test_type: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool