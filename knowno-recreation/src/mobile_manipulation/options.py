import re
from typing import List, Tuple
from src.llm_client import LLMClient

FOUR_CHOICES_RE = re.compile(
    r'^\s*A\.\s*(.+?)\s*\n\s*B\.\s*(.+?)\s*\n\s*C\.\s*(.+?)\s*\n\s*D\.\s*(.+?)\s*$',
    flags=re.DOTALL | re.IGNORECASE
)

def parse_four(text: str) -> List[str]:
    m = FOUR_CHOICES_RE.search(text.strip())
    if not m:
        return []
    return [m.group(i).strip() for i in range(1, 5)]

class OptionGenerator:
    def __init__(self, template: str, llm: LLMClient | None = None):
        self.template = template
        self.llm = llm or LLMClient()

    def generate(self, situation: str, retry: int = 2) -> List[str]:
        prompt = self.template.replace("{SITUATION}", situation).strip() + "\n"
        from requests import post
        import os, json

        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        url  = base + "/api/generate"
        payload = {
            "model": self.llm.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 128}
        }
        for _ in range(retry + 1):
            r = post(url, json=payload, timeout=120)
            r.raise_for_status()
            js = r.json()
            txt = js.get("response", "").strip()
            opts = parse_four(txt)
            if len(opts) == 4:
                return opts
        return []  
