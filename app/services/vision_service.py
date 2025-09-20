import requests


class VisionService:
    def __init__(self, model_id: str, hf_token: str):
        self.url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    def analyze_image_url(self, image_url: str) -> dict:
        # Many HF vision models accept URL via JSON payload; otherwise download bytes and send as binary.
        payload = {"inputs": image_url}
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}