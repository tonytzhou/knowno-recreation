import json
import argparse
from pathlib import Path

from src.llm_client import LLMClient
from src.cp import nonconf_correct_from_probs, split_cp_quantile
from src.prompt_utils import build_mcqa_prompt

LETTERS = ["A", "B", "C", "D", "E"]

def letter_from_index(i: int) -> str:
    return LETTERS[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Alpha levels for split-CP (e.g. --alphas 0.25 0.42 0.15 0.24)",
    )
    ap.add_argument(
        "--calib_path",
        type=str,
        default="data/calib.jsonl",
        help="Path to calibration jsonl (one item per line).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of calibration items (0 = use all).",
    )
    args = ap.parse_args()

    with open(args.calib_path, "r", encoding="utf-8") as f:
        calib_rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        calib_rows = calib_rows[:args.limit]

    cli = LLMClient()
    nonconfs = []  

    for ex in calib_rows:
        context = ex["context"]
        options = ex["options"]
        correct_letter = letter_from_index(ex["correct_index"])

        prompt = build_mcqa_prompt(context, options)
        probs  = cli.get_option_probs(prompt)  # dict over A..E ~sum to 1

        s = nonconf_correct_from_probs(probs, correct_letter)
        nonconfs.append(s)

    thresholds = {f"{a:.6g}": float(split_cp_quantile(nonconfs, a)) for a in args.alphas}

    Path("artifacts").mkdir(exist_ok=True)
    out_path = Path("artifacts/cp_thresholds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Saved: {out_path} {thresholds} (calib_used={len(nonconfs)})")

if __name__ == "__main__":
    main()
