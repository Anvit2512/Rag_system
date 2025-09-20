import os, requests


class LLMService:
    def __init__(self, model_id: str, hf_token: str):
        self.model_id = model_id
        self.hf_token = hf_token
        self.url = f"https://api-inference.huggingface.co/models/{model_id}"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
        r = requests.post(self.url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()
        # HF text models often return a list of dicts
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"]
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        # Fallback: try to stringify
        return str(out)