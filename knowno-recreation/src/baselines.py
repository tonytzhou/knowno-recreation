import argparse, json, math, random, re
from collections import Counter, defaultdict
from typing import Dict, List

ASK_HELP = "ASK_HELP"
OPT_KEYS = ["A", "B", "C", "D", "E"]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sorted_options_by_scores(options_dict, scores_dict):
    items = []
    order_index = {k: i for i, k in enumerate(OPT_KEYS)}
    for k in options_dict.keys():
        score = float(scores_dict.get(k, float("-inf")))
        items.append((k, score))
    items.sort(key=lambda kv: (-kv[1], order_index.get(kv[0], 999)))
    return [k for k, _ in items]


def _top1(options_dict, scores_dict):
    ranked = _sorted_options_by_scores(options_dict, scores_dict)
    return ranked[0] if ranked else None


def _looks_like_log(values: List[float]) -> bool:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return False
    frac_nonpos = sum(v <= 0 for v in finite) / len(finite)
    return (frac_nonpos > 0.6) and ((max(finite) - min(finite)) > 2.0)


def _normalize_scores(scores_dict: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    vals = [float(scores_dict.get(k, float("-inf"))) for k in keys]
    if not any(math.isfinite(v) for v in vals):
        return {k: 0.0 for k in keys}
    if _looks_like_log(vals):
        m = max(v for v in vals if math.isfinite(v))
        exps = [math.exp((v if math.isfinite(v) else -1e12) - m) for v in vals]
        Z = sum(exps)
        return {k: (exps[i] / Z if Z > 0 else 0.0) for i, k in enumerate(keys)}
    clipped = [max(0.0, (v if math.isfinite(v) else 0.0)) for v in vals]
    Z = sum(clipped)
    return {k: (clipped[i] / Z if Z > 0 else 0.0) for i, k in enumerate(keys)}


def _get_option_scores(ex, option_keys):
    for key in ["probs", "scores", "option_probs", "logprobs", "option_logps", "probs_logits"]:
        if key in ex and isinstance(ex[key], dict) and any(k in ex[key] for k in option_keys):
            return {k: ex[key].get(k, 0.0) for k in option_keys}
    if "probs_runs" in ex and isinstance(ex["probs_runs"], list) and ex["probs_runs"]:
        acc = {k: 0.0 for k in option_keys}; n = 0
        for run in ex["probs_runs"]:
            if not isinstance(run, dict): continue
            vals = {k: float(run.get(k, 0.0)) for k in option_keys}
            probs = _normalize_scores(vals, option_keys)
            if sum(probs.values()) > 0:
                for k in option_keys: acc[k] += probs[k]
                n += 1
        if n > 0:
            return {k: acc[k] / n for k in option_keys}
    return {k: 0.0 for k in option_keys}


def _build_set_cum_threshold(options_dict, probs_dict, thresh: float) -> List[str]:
    mass = sum(max(0.0, float(probs_dict.get(k, 0.0))) for k in options_dict.keys())
    if mass <= 0: return []
    ranked = _sorted_options_by_scores(options_dict, probs_dict)
    pred, cum = [], 0.0
    for k in ranked:
        pred.append(k)
        cum += float(probs_dict.get(k, 0.0))
        if cum >= thresh: break
    if not pred and ranked: pred = [ranked[0]]
    return pred


def _counts_from_runs(ex):
    if "ensemble_counts" in ex and isinstance(ex["ensemble_counts"], dict):
        return Counter({k: int(v) for k, v in ex["ensemble_counts"].items()})
    if "samples" in ex and isinstance(ex["samples"], list):
        return Counter(ex["samples"])
    if "probs_runs" in ex and isinstance(ex["probs_runs"], list):
        c = Counter()
        for run_scores in ex["probs_runs"]:
            if not isinstance(run_scores, dict): continue
            top = _top1(ex["options"], run_scores)
            if top: c[top] += 1
        return c
    return Counter()


def _normalize_counts_to_freq(counts, keys):
    tot = sum(counts.values())
    if tot <= 0: return {k: 0.0 for k in keys}
    return {k: counts.get(k, 0) / tot for k in keys}


def _parse_prompt_set(ex):
    if "prompt_set" in ex and isinstance(ex["prompt_set"], list):
        return [s for s in ex["prompt_set"] if isinstance(s, str) and s in OPT_KEYS]
    txt = ex.get("prompt_set_text")
    if not isinstance(txt, str): return []
    m = re.search(r"\[(.*?)\]", txt)
    if not m: return []
    out = []
    for t in [t.strip() for t in m.group(1).split(",")]:
        t = t.strip().strip("'\"")
        if t in OPT_KEYS: out.append(t)
    return out


def _is_certain(val):
    if isinstance(val, bool): return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("certain","sure","confident","yes","y","1","true"): return True
        if s in ("uncertain","not sure","no","n","0","false"): return False
    return False


def _needed_cumprob_to_hit_any_correct(ex, probs: Dict[str, float]) -> float:
    cp = set(ex.get("cp_set") or [])
    if not cp: return 1.0
    ranked = _sorted_options_by_scores(ex["options"], probs)
    cum = 0.0
    for k in ranked:
        cum += float(probs.get(k, 0.0))
        if k in cp: return cum
    return 1.0


def _conformal_threshold_simple(cal_examples: List[dict], epsilon: float) -> float:
    vals = []
    for ex in cal_examples:
        keys = list(ex["options"].keys())
        probs = _normalize_scores(_get_option_scores(ex, keys), keys)
        vals.append(_needed_cumprob_to_hit_any_correct(ex, probs))
    if not vals: return 1.0 - epsilon
    vals.sort()
    n = len(vals)
    k = min(max(1, math.ceil((n + 1) * (1.0 - epsilon))), n)
    return float(vals[k - 1])


def _conformal_threshold_knowno(cal_examples: List[dict], epsilon: float) -> float:
    max_correct = []
    for ex in cal_examples:
        keys = list(ex["options"].keys())
        probs = _normalize_scores(_get_option_scores(ex, keys), keys)
        cp = set(ex.get("cp_set") or [])
        if not cp:
            continue
        s_star = max((probs.get(k, 0.0) for k in cp), default=0.0)
        max_correct.append(s_star)
    if not max_correct:
        return 1.0 - epsilon  # fallback
    max_correct.sort()
    n = len(max_correct)
    k = min(max(1, math.ceil((n + 1) * epsilon)), n)  # ε-quantile with correction
    return float(max_correct[k - 1])


def build_prediction_set(ex, baseline, epsilon,
                         simple_tau=None,
                         knowno_tau=None):
    target_cov = 1.0 - float(epsilon)
    options = ex["options"]

    if baseline == "simple":
        keys = list(options.keys())
        probs_used = _normalize_scores(_get_option_scores(ex, keys), keys)
        thresh = simple_tau if simple_tau is not None else target_cov
        pred_set = _build_set_cum_threshold(options, probs_used, thresh)
        if not pred_set: return [], ASK_HELP, True, None
        top = _top1({k: options[k] for k in pred_set}, {k: probs_used.get(k, 0.0) for k in pred_set})
        return pred_set, top, False, probs_used

    if baseline == "ensemble":
        keys = list(options.keys())
        counts = _counts_from_runs(ex)
        if counts:
            freqs = _normalize_counts_to_freq(counts, keys)
            pred_set = _build_set_cum_threshold(options, freqs, target_cov)
            if not pred_set: return [], ASK_HELP, True, None
            top = _top1({k: options[k] for k in pred_set}, {k: freqs.get(k, 0.0) for k in pred_set})
            return pred_set, top, False, freqs
        return build_prediction_set(ex, "simple", epsilon, simple_tau=None)

    if baseline == "prompt":
        pred_set = _parse_prompt_set(ex)
        if not pred_set: return [], ASK_HELP, True, None
        ordered = [k for k in OPT_KEYS if k in pred_set]
        chosen = ordered[0] if ordered else pred_set[0]
        return pred_set, chosen, False, None

    if baseline == "binary":
        if not _is_certain(ex.get("binary_uncertainty")):
            return [], ASK_HELP, True, None
        if "top1" in ex and isinstance(ex["top1"], str):
            top = ex["top1"]; return [top], top, False, None
        keys = list(options.keys())
        probs_used = _normalize_scores(_get_option_scores(ex, keys), keys)
        top = _top1(options, probs_used)
        if top is None: return [], ASK_HELP, True, None
        return [top], top, False, probs_used

    if baseline == "nohelp":
        keys = list(options.keys())
        probs_used = _normalize_scores(_get_option_scores(ex, keys), keys)
        top = _top1(options, probs_used)
        if top is None: return [], ASK_HELP, True, None
        return [top], top, False, probs_used

    if baseline == "knowno":
        keys = list(options.keys())
        probs_used = _normalize_scores(_get_option_scores(ex, keys), keys)
        tau = 0.0 if knowno_tau is None else float(knowno_tau)
        pred = [k for k in keys if probs_used.get(k, 0.0) >= tau]
        pred.sort(key=lambda k: (-probs_used.get(k, 0.0), OPT_KEYS.index(k) if k in OPT_KEYS else 999))
        if not pred: return [], ASK_HELP, True, None
        top = pred[0]
        return pred, top, False, probs_used

    raise ValueError(f"Unknown baseline: {baseline}")


def merge_situations_with_scores(situations_path, scores_path):
    sits = list(read_jsonl(situations_path))
    scs = list(read_jsonl(scores_path))
    if sits and scs and all("id" in x for x in sits) and all("id" in x for x in scs):
        by_id = {s["id"]: s for s in scs}
        return [{**ex, **by_id.get(ex["id"], {})} for ex in sits]
    n = min(len(sits), len(scs))
    merged = [{**sits[i], **scs[i]} for i in range(n)]
    if len(sits) > n:
        merged.extend(sits[n:])
    return merged


def summarize(examples, baseline, epsilon, target, label,
              simple_tau=None, knowno_tau=None):
    N = 0
    help_cnt = 0
    set_size_sum = 0.0
    plan_succ = 0
    task_succ = 0
    have_plan = False
    have_task = False
    top_logps = []
    chosen_logps = []

    for ex in examples:
        if "options" not in ex:  continue
        N += 1
        pred, chosen, _, probs_used = build_prediction_set(
            ex, baseline, epsilon, simple_tau=simple_tau, knowno_tau=knowno_tau
        )
        set_size_sum += len(pred)

        if baseline in {"simple", "ensemble", "prompt", "knowno"}:
            help_cnt += int(len(pred) > 1)
        elif baseline == "binary":
            help_cnt += int(chosen == ASK_HELP)

        cp = set(ex.get("cp_set") or [])
        if cp:
            have_plan = True
            if set(pred).intersection(cp): plan_succ += 1
            have_task = True
            if isinstance(chosen, str) and chosen in cp: task_succ += 1

        if isinstance(probs_used, dict) and probs_used:
            top_key = max(probs_used, key=lambda k: probs_used[k])
            p_top = max(1e-12, float(probs_used.get(top_key, 0.0)))
            top_logps.append(math.log(p_top))
            if isinstance(chosen, str):
                p_ch = max(1e-12, float(probs_used.get(chosen, 0.0)))
                chosen_logps.append(math.log(p_ch))

    task_rate = (task_succ / N) if (N and have_task) else 0.0
    dev = abs(task_rate - target)

    out = {
        "Label": label,
        "N": N,
        "Plan Succ": (plan_succ / N) if (N and have_plan) else "-",
        "Task Succ": (task_succ / N) if (N and have_task) else "-",
        "Dev from (1-eps)": dev if N else "-",
        "Set Size": (set_size_sum / N) if N else 0.0,
        "Help": (help_cnt / N) if N else 0.0,
        "Avg Top LogP": (sum(top_logps) / len(top_logps)) if top_logps else "-",
        "Avg Chosen LogP": (sum(chosen_logps) / len(chosen_logps)) if chosen_logps else "-",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", choices=["simple", "ensemble", "prompt", "binary", "nohelp", "knowno"], required=True)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--simple_cp", action="store_true", help="Enable split-conformal calibration for Simple Set")
    ap.add_argument("--cal_ratio", type=float, default=0.30, help="Fraction of labeled data for calibration")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--knowno_tau", type=float, default=None,
                    help="Use a fixed τ_K (skip calibration) if provided")
    ap.add_argument("--knowno_cp", action="store_true",
                    help="Calibrate τ_K from a split of the data (recommended)")

    args = ap.parse_args()
    target = 1.0 - float(args.epsilon)
    data = merge_situations_with_scores(args.situations, args.scores)

    rng = random.Random(args.seed)
    labeled = [ex for ex in data if "options" in ex and ex.get("cp_set")]

    simple_tau = None
    knowno_tau = args.knowno_tau
    cal_info = ""

    if args.baseline == "simple" and args.simple_cp:
        if labeled:
            idx = list(range(len(labeled))); rng.shuffle(idx)
            m = max(1, int(len(idx) * max(0.0, min(1.0, args.cal_ratio))))
            cal = [labeled[i] for i in idx[:m]]
            simple_tau = _conformal_threshold_simple(cal, args.epsilon)
            cal_info = f" (Simple CP τ={simple_tau:.3f}, n_cal={len(cal)})"

    if args.baseline == "knowno":
        if knowno_tau is None: 
            if not args.knowno_cp:
                raise SystemExit("For baseline=knowno, pass --knowno_tau or enable --knowno_cp to calibrate τ_K.")
            if labeled:
                idx = list(range(len(labeled))); rng.shuffle(idx)
                m = max(1, int(len(idx) * max(0.0, min(1.0, args.cal_ratio))))
                cal = [labeled[i] for i in idx[:m]]
                knowno_tau = _conformal_threshold_knowno(cal, args.epsilon)
                cal_info = f" (KnowNo τ_K={knowno_tau:.3f}, n_cal={len(cal)})"
            else:
                knowno_tau = 1.0 - args.epsilon
                cal_info = " (KnowNo τ_K fallback)"

    groups = defaultdict(list)
    for ex in data:
        groups[ex.get("setting", "Unknown")].append(ex)

    overall = summarize(data, args.baseline, args.epsilon, target, "OVERALL",
                        simple_tau=simple_tau, knowno_tau=knowno_tau)
    per_setting = [summarize(v, args.baseline, args.epsilon, target, k,
                             simple_tau=simple_tau, knowno_tau=knowno_tau)
                   for k, v in sorted(groups.items())]

    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"== {args.baseline}{cal_info} @ target (1-ε)={target:.2f} ==")
    headers = ["Label", "N", "Plan Succ", "Task Succ", "Dev from (1-eps)", "Set Size", "Help", "Avg Top LogP", "Avg Chosen LogP"]
    print(" | ".join(f"{h:>16s}" for h in headers))
    def row(d): return " | ".join(f"{fmt(d[h]):>16s}" for h in headers)
    print(row(overall))
    for d in per_setting:
        print(row(d))


if __name__ == "__main__":
    main()
