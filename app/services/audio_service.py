import requests


class AudioService:
    def __init__(self, model_id: str, hf_token: str):
        self.url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    def analyze_audio_url(self, audio_url: str) -> dict:
        payload = {"inputs": audio_url}
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()