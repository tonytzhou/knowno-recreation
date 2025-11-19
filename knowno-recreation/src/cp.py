import math
from typing import List, Dict

EPS = 1e-12

def nonconf_correct_from_probs(probs: Dict[str, float], correct_letter: str) -> float:
    p = max(float(probs.get(correct_letter, 0.0)), EPS)
    return -math.log(p)

def split_cp_quantile(nonconfs: List[float], alpha: float) -> float:
    if not nonconfs:
        raise ValueError("Empty calibration set")
    vals = sorted(float(x) for x in nonconfs)
    n = len(vals)
    k = math.ceil((n + 1) * (1.0 - alpha)) - 1
    k = min(max(k, 0), n - 1)
    return vals[k]
