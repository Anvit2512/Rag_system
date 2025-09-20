from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    audio_url: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    citations: List[str] = []
    images: List[str] = []