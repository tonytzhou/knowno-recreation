import math
from typing import Dict, List

LETTERS = ["A", "B", "C", "D", "E"]
EPS = 1e-12

def probs_to_nonconfs(probs: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, p in probs.items():
        out[k] = -math.log(max(float(p), EPS))
    return out

def cp_set(probs: Dict[str, float], q: float) -> List[str]:
    ncs = probs_to_nonconfs(probs)
    return [k for k, s in ncs.items() if s <= q]

def decision_from_set(S: List[str]) -> str:
    if len(S) == 1:
        return S[0]
    return "ASK_HELP"
