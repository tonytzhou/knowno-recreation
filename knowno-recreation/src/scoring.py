import math
from typing import Dict

def option_nonconformity(p: float) -> float:
    return -math.log(max(min(p, 1.0), 1e-3))

def probs_to_nonconfs(probs: Dict[str, float]) -> Dict[str, float]:
    return {k: option_nonconformity(v) for k, v in probs.items()}
