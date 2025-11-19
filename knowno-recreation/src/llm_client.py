import os, math, json, requests
from typing import Dict, List, Optional

ROOT_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TIMEOUT  = float(os.getenv("OLLAMA_TIMEOUT", "120"))
DEBUG    = os.getenv("OLLAMA_DEBUG", "") != ""
LETTERS  = ["A","B","C","D","E"]

def _softmax(lps: Dict[str, float]) -> Dict[str, float]:
    m = max(lps.values())
    exps = {k: math.exp(v - m) for k, v in lps.items()}
    Z = sum(exps.values()) or 1.0
    return {k: exps[k] / Z for k in lps}

def _smooth(p: Dict[str, float], eps: float = 5e-3) -> Dict[str, float]:
    q = {k: p.get(k, 0.0) + eps for k in LETTERS}
    Z = sum(q.values()) or 1.0
    return {k: v / Z for k, v in q.items()}

def _dbg(label: str, obj):
    if DEBUG:
        print(f"\n--- DEBUG {label} ---")
        try:
            print(json.dumps(obj, indent=2)[:2000])
        except Exception:
            print(str(obj)[:2000])

class LLMClient:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or MODEL
        self.temperature = float(temperature)
        self.v1 = ROOT_URL + "/v1"

    def build_prompt(self, context: str, options: List[str], template: str) -> str:
        A,B,C,D,E = options
        base = (template.replace("{CONTEXT}", context)
                        .replace("{A}", A).replace("{B}", B)
                        .replace("{C}", C).replace("{D}", D)
                        .replace("{E}", E))
        return base.rstrip() + "\n\nAnswer: "

    def _post(self, url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        if DEBUG: _dbg("REQ "+url, payload)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"{e}\nURL: {url}\nPAYLOAD: {payload}\nRESPONSE: {r.text}") from None
        js = r.json()
        if DEBUG: _dbg("RESP "+url, js)
        return js

    def _pick_from_topmap_objlist(self, objlist: List[dict]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for ent in objlist:
            tok = (ent.get("token") or "").strip()
            lp  = ent.get("logprob")
            if lp is None: continue
            for L in LETTERS:
                if tok == L or tok.startswith(L) or tok.startswith(f"({L}"):
                    out[L] = float(lp) if (L not in out or float(lp) > out[L]) else out[L]
        return out

    def _parse_native(self, resp: dict) -> Optional[Dict[str, float]]:
        lp_list = resp.get("logprobs")
        if isinstance(lp_list, list) and lp_list:
            first = lp_list[0]
            top = first.get("top_logprobs")
            if isinstance(top, list) and top:
                return self._pick_from_topmap_objlist(top)

            tok = (first.get("token") or "").strip()
            lp  = first.get("logprob")
            if lp is not None and tok:
                out = {k: -50.0 for k in LETTERS}
                for L in LETTERS:
                    if tok == L or tok.startswith(L) or tok.startswith(f"({L}"):
                        out[L] = float(lp)
                        return out
        toks = resp.get("tokens")
        if isinstance(toks, list) and toks:
            t0 = toks[0]
            top = t0.get("top_logprobs")
            if isinstance(top, list) and top:
                return self._pick_from_topmap_objlist(top)
            tok = (t0.get("token") or "").strip()
            lp  = t0.get("logprob")
            if lp is not None and tok:
                out = {k: -50.0 for k in LETTERS}
                for L in LETTERS:
                    if tok == L or tok.startswith(L) or tok.startswith(f"({L}"):
                        out[L] = float(lp)
                        return out
        return None

    def _native_next_logprobs(self, prompt: str) -> Optional[Dict[str, float]]:
        p1 = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1,
                "logprobs": True,
                "top_logprobs": 5
            }
        }
        resp = self._post(f"{ROOT_URL}/api/generate", p1)
        out = self._parse_native(resp)
        if out: return out

        p2 = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "num_predict": 1,
            "logprobs": True,
            "top_logprobs": 5
        }
        resp = self._post(f"{ROOT_URL}/api/generate", p2)
        return self._parse_native(resp)

    def _v1_toplogprobs(self, prompt: str) -> Optional[Dict[str, float]]:
        resp = self._post(
            f"{self.v1}/completions",
            {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "echo": False, "logprobs": 50}
        )
        ch  = resp["choices"][0]
        lps = ch.get("logprobs") or {}
        arr = lps.get("top_logprobs")
        if not isinstance(arr, list) or not arr or not isinstance(arr[0], dict):
            return None
        out: Dict[str, float] = {}
        for tok, lpv in arr[0].items():
            s = (tok or "").strip()
            for L in LETTERS:
                if s == L or s.startswith(L) or s.startswith(f"({L}"):
                    out[L] = float(lpv) if (L not in out or float(lpv) > out[L]) else out[L]
        return out if out else None

    def _v1_score_tokens(self, prompt: str) -> dict:
        return self._post(
            f"{self.v1}/completions",
            {"model": self.model, "prompt": prompt, "max_tokens": 0, "temperature": 0.0, "echo": True, "logprobs": 5}
        )

    def _v1_base_len(self, prompt: str) -> int:
        resp = self._v1_score_tokens(prompt)
        lp = (resp["choices"][0].get("logprobs") or {})
        toks = lp.get("tokens") or []
        lps  = lp.get("token_logprobs") or []
        if not toks or not lps or len(toks) != len(lps):
            raise RuntimeError("No echo token arrays from /v1/completions")
        return len(toks)

    def _v1_letter_lp_by_index(self, prompt: str, base_len: int, L: str) -> float:
        resp = self._v1_score_tokens(prompt + L)
        lp = (resp["choices"][0].get("logprobs") or {})
        lps = lp.get("token_logprobs") or []
        if not lps:
            return -50.0
        val = lps[-1]
        return float(val) if val is not None else -50.0

    def get_option_probs(self, prompt: str) -> Dict[str, float]:
        # Native first
        nat = self._native_next_logprobs(prompt)
        if nat:
            return _smooth(_softmax({k: nat.get(k, -50.0) for k in LETTERS}), eps=5e-3)

        try:
            top = self._v1_toplogprobs(prompt)
            if top:
                return _smooth(_softmax({k: top.get(k, -50.0) for k in LETTERS}), eps=5e-3)
        except Exception:
            pass

        try:
            base = self._v1_base_len(prompt)
            lps = {L: self._v1_letter_lp_by_index(prompt, base, L) for L in LETTERS}
            return _smooth(_softmax(lps), eps=5e-3)
        except Exception:
            return {k: 1.0/5 for k in LETTERS}
