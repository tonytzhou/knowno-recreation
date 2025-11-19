import argparse
import json
import random
import pathlib
from typing import List

from src.llm_client import LLMClient
from src.mobile_manipulation.options import OptionGenerator
from src.mobile_manipulation.cp_runtime import cp_set, decision_from_set


ROOT    = pathlib.Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
ART     = ROOT / "artifacts"

E_TEXT  = "An option not listed here."
LETTERS = ["A", "B", "C", "D", "E"]


def load_threshold(alpha: float) -> float:
    th = json.loads((ART / "cp_thresholds.json").read_text())
    key = str(alpha)
    if key not in th:
        fmt = f"{alpha:.2f}".rstrip("0").rstrip(".")
        key = fmt if fmt in th else key
    if key not in th:
        raise ValueError(f"alpha {alpha} not in thresholds {list(th.keys())}")
    return float(th[key])


def build_mcqa_prompt(context: str, options5: List[str], mcqa_template: str) -> str:
    A, B, C, D, E = options5
    filled = (
        mcqa_template
        .replace("{CONTEXT}", context)
        .replace("{A}", A).replace("{B}", B)
        .replace("{C}", C).replace("{D}", D)
        .replace("{E}", E)
    )
    return filled.rstrip() + "\n\nAnswer: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", type=str, required=True, help="jsonl with {'situation': str}")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=str, default=str(ROOT / "out/predictions.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="print per-situation details")
    ap.add_argument("--progress_every", type=int, default=0,
                    help="if >0, print a progress line every N items (still quieter than --verbose)")
    args = ap.parse_args()

    random.seed(args.seed)

    opt_tmpl = (PROMPTS / "option_gen_template.txt").read_text()
    mcqa_template = (
        "You are a careful planner that selects exactly ONE best next step.\n\n"
        "Context:\n{CONTEXT}\n\n"
        "Options:\n"
        "A. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\n\n"
        "Question: Which option is correct? Answer with just the single letter (A, B, C, D, or E)."
    )

    q    = load_threshold(args.alpha)
    llm  = LLMClient()
    optg = OptionGenerator(opt_tmpl, llm=llm)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.situations, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            sit = rec["situation"].strip()

            ad = optg.generate(sit)
            if len(ad) != 4:
                ad = ["Ask for clarification.", "Do nothing.", "Take any option.", "Stop."]

            options5 = list(ad)
            insert_idx = random.randrange(5)
            options5.insert(insert_idx, E_TEXT)

            prompt = build_mcqa_prompt(sit, options5, mcqa_template)
            probs  = llm.get_option_probs(prompt)  # dict like {"A": pA, ...}
            S      = cp_set(probs, q)
            decision = decision_from_set(S)
            clarify = "Which option do you mean — A, B, C, D, or E?" if decision == "ASK_HELP" else None

            out_record = {
                "situation": sit,
                "options": {LETTERS[j]: options5[j] for j in range(5)},
                "probs": probs,
                "q": q,
                "cp_set": S,
                "decision": decision,
                "clarify": clarify,
                "insert_idx_for_E": insert_idx
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            count += 1

            if args.verbose:
                print(f"\n=== Situation {count} ===")
                for j, L in enumerate(LETTERS):
                    print(f"{L}. {options5[j]}")
                print("probs:", probs)
                print("CP set:", S, " decision:", decision)
                if clarify:
                    print("Ask for help:", clarify)
            elif args.progress_every and (count % args.progress_every == 0):
                print(f"[progress] processed {count} items...")

    print(f"Saved {count} predictions to {out_path}")

if __name__ == "__main__":
    main()
