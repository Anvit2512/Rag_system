import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest, ChatResponse
from .orchestrator import ConversationOrchestrator
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(title="RAG Real-Time API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)


orch = ConversationOrchestrator()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message, citations, images = orch.handle_chat(req.text, req.image_url, req.audio_url)
    return ChatResponse(message=message, citations=citations, images=images)


@app.get("/health")
async def health():
    return {"status": "ok"}