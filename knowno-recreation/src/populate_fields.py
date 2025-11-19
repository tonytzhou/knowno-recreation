import argparse, json, random, math, sys
from collections import Counter, OrderedDict

OPT_KEYS = ["A","B","C","D","E"]

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_probs(options, probs):
    dense = {k: float(probs.get(k, 0.0)) for k in options.keys()}
    vals = [max(0.0, v) for v in dense.values()]
    s = sum(vals)
    if s <= 0:
        u = 1.0 / max(1, len(options))
        return {k: u for k in options.keys()}
    return {k: max(0.0, v)/s for k, v in dense.items()}

def sorted_by_score_desc(options, scores_dict):
    order_index = {k:i for i,k in enumerate(OPT_KEYS)}
    items = []
    for k in options.keys():
        v = float(scores_dict.get(k, float("-inf")))
        items.append((k, v))
    items.sort(key=lambda kv: (-kv[1], order_index.get(kv[0], 999)))
    return [k for k,_ in items]

def top1(options, scores):
    ranked = sorted_by_score_desc(options, scores)
    return ranked[0] if ranked else None

def cumulative_set(options, probs, target_cov):
    ranked = sorted_by_score_desc(options, probs)
    cum, out = 0.0, []
    for k in ranked:
        out.append(k)
        cum += probs[k]
        if cum >= target_cov:
            break
    if not out and ranked:
        out = [ranked[0]]
    return out

def sample_from_probs(options, probs, n=1):
    keys = list(options.keys())
    weights = [probs[k] for k in keys]
    cw = []
    acc = 0.0
    for w in weights:
        acc += w
        cw.append(acc)
    draws = []
    for _ in range(n):
        r = random.random() * cw[-1]
        for i, c in enumerate(cw):
            if r <= c:
                draws.append(keys[i])
                break
    return draws

def merge_rows(situations, scores):
    use_by_text = False
    if all(isinstance(x.get("situation"), str) for x in situations) and \
       all(isinstance(x.get("situation"), str) for x in scores):
        sit_map = {}
        ok = True
        for i, ex in enumerate(situations):
            s = ex.get("situation")
            if s in sit_map: ok = False; break
            sit_map[s] = i
        if ok:
            use_by_text = True
            merged = []
            for sc in scores:
                s = sc.get("situation")
                if s in sit_map:
                    base = dict(situations[sit_map[s]])
                else:
                    base = {}
                base.update(sc)
                merged.append(base)
            return merged

    n = min(len(situations), len(scores))
    merged = []
    for i in range(n):
        base = dict(situations[i])
        base.update(scores[i])
        merged.append(base)
    # append any extras
    for i in range(n, len(situations)):
        merged.append(dict(situations[i]))
    for i in range(n, len(scores)):
        merged.append(dict(scores[i]))
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", required=True, help="data/situations.jsonl")
    ap.add_argument("--scores", required=True, help="out/preds.jsonl (has probs, cp_set, etc.)")
    ap.add_argument("--out", required=True, help="augmented jsonl")
    ap.add_argument("--epsilon", type=float, default=0.15, help="used ONLY for the Binary certainty rule via Simple Set")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for ensemble/prompt sampling")
    ap.add_argument("--ensemble_trials", type=int, default=20, help="number of trials for Ensemble Set")
    ap.add_argument("--prompt_u_min", type=float, default=0.50, help="min random target for prompt-set cumulative")
    ap.add_argument("--prompt_u_max", type=float, default=0.95, help="max random target for prompt-set cumulative")
    args = ap.parse_args()

    random.seed(args.seed)

    situations = read_jsonl(args.situations)
    scores      = read_jsonl(args.scores)
    rows        = merge_rows(situations, scores)

    out_rows = []
    for ex in rows:
        options = ex.get("options")
        if not isinstance(options, dict) or not options:
            out_rows.append(ex)
            continue

        probs_raw = ex.get("probs", {})
        probs = normalize_probs(options, probs_raw)

        if "ensemble_counts" not in ex and "samples" not in ex and "probs_runs" not in ex:
            draws = sample_from_probs(options, probs, n=args.ensemble_trials)
            cnts = Counter(draws)
            ex["samples"] = draws
            ex["ensemble_counts"] = dict(cnts)


        if "prompt_set" not in ex and "prompt_set_text" not in ex:
            U = random.uniform(args.prompt_u_min, args.prompt_u_max)
            pred_set = cumulative_set(options, probs, U)
            ex["prompt_set"] = pred_set
            ex["prompt_set_text"] = f"Prediction set: [{', '.join(pred_set)}]"

        if "binary_uncertainty" not in ex:
            target_cov = 1.0 - float(args.epsilon)
            simple_set = cumulative_set(options, probs, target_cov)
            certain = (len(simple_set) <= 1)
            ex["binary_uncertainty"] = "Certain" if certain else "Uncertain"
            if "top1" not in ex:
                ex["top1"] = top1(options, probs)

        out_rows.append(ex)

    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} examples -> {args.out}")

if __name__ == "__main__":
    main()
