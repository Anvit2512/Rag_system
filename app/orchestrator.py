import os, requests
from .services.vision_service import VisionService
from .services.audio_service import AudioService
from .services.llm_service import LLMService
from .data.vector_store import VectorStore
from .data.metadata_store import MetadataStore


class ConversationOrchestrator:
    def __init__(self):
        token = os.getenv("HF_API_TOKEN")
        self.vdb = VectorStore(os.getenv("VDB_DIR"), os.getenv("HF_EMBED_MODEL_ID"), token)
        self.meta = MetadataStore(os.getenv("SQLITE_URL"))
        self.llm = LLMService(os.getenv("HF_LLM_MODEL_ID"), token)
        self.vision = VisionService(os.getenv("HF_VISION_MODEL_ID"), token)
        self.audio = AudioService(os.getenv("HF_AUDIO_MODEL_ID"), token)
        self.files_dir = os.getenv("FILES_DIR", "./data/files")
        self.embed_model_id = os.getenv("HF_EMBED_MODEL_ID")
        self.hf_token = token
        self.max_ctx = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))


    # def _hf_embed(self, text: str) -> list[float]:
    #     url = f"https://api-inference.huggingface.co/models/{self.embed_model_id}"
    #     headers = {"Authorization": f"Bearer {self.hf_token}"}
    #     payload = {"inputs": text}
    #     r = requests.post(url, headers=headers, json=payload, timeout=60)
    #     r.raise_for_status()
    #     vec = r.json()
    #     # Some embedding endpoints return {"embeddings": [[...]]} or [[...]]
    #     if isinstance(vec, dict) and "embeddings" in vec:
    #         return vec["embeddings"][0]
    #     if isinstance(vec, list):
    #         return vec[0] if vec and isinstance(vec[0], list) else vec
    #     raise ValueError("Unexpected embedding response")
    def _hf_embed(self, text: str) -> list[float]:
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.embed_model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        payload = {"inputs": text.strip(), "options": {"wait_for_model": True}}
        import time, requests
        for attempt in range(5):
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(2 + attempt * 2); continue
            if r.status_code == 400:
                print("HF 400 body:", r.text[:300])
            r.raise_for_status()
            js = r.json()
            if isinstance(js, list) and js and isinstance(js[0], (float, int)):
                return js
            if isinstance(js, list) and js and isinstance(js[0], list):
                return js[0]
            if isinstance(js, list) and js and isinstance(js[0], dict) and "embedding" in js[0]:
                return js[0]["embedding"]
            raise ValueError(f"Unexpected embedding response: {str(js)[:200]}")
        raise RuntimeError("Embedding endpoint busy after retries")



    def _build_rag_prompt(self, question: str, ctx_docs: list[str]) -> str:
        context = "\n\n".join(f"[Doc {i+1}] {t}" for i, t in enumerate(ctx_docs))
        return (
            "You are a precise assistant for washing machine manuals. Use the provided context to answer.\n"
            "If the answer isn't in context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )


    def handle_chat(self, text: str | None, image_url: str | None, audio_url: str | None):
        vision_json, audio_json = None, None
        citations, images = [], []

        # Optional modalities
        if image_url:
            vision_json = self.vision.analyze_image_url(image_url)
            images.append(image_url)
        if audio_url:
            audio_json = self.audio.analyze_audio_url(audio_url)

        # RAG: embed question → query VDB → fetch metadata texts
        q = text or "Describe the image/audio findings and answer the user succinctly."
        q_vec = self._hf_embed(q)
        qres = self.vdb.query(q_vec, n_results=self.max_ctx)
        ids = qres.get("ids", [[]])[0]
        docs = qres.get("documents", [[]])[0]
        found = self.meta.get_texts_by_ids(ids) if ids else []
        citations = [d.id for d in found]
        ctx_texts = [d.text for d in found] if found else docs

        # Compose prompt with structured tool outputs
        extras = []
        if vision_json:
            extras.append(f"Vision Findings: {vision_json}")
        if audio_json:
            extras.append(f"Audio Findings: {audio_json}")
        extra_block = "\n\n" + "\n".join(extras) if extras else ""

        prompt = self._build_rag_prompt(q, ctx_texts) + extra_block
        answer = self.llm.generate(prompt)
        return answer, citations, images