import json, os, random, pathlib, numpy as np
from src.llm_client import LLMClient
from src.scoring import probs_to_nonconfs

DATA   = pathlib.Path("data")
PROMPTS= pathlib.Path("prompts")
ART    = pathlib.Path("artifacts")

tmpl = (PROMPTS/"mcqa_template.txt").read_text()
thresholds = json.loads(open(ART/"cp_thresholds.json").read())
client = LLMClient()

TEST_LIMIT   = int(os.getenv("TEST_LIMIT", "0"))   # 0 = use all
TEST_SHUFFLE = os.getenv("TEST_SHUFFLE", "1") != "0"
RAND_SEED    = int(os.getenv("TEST_SEED", "42"))
random.seed(RAND_SEED)

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

all_test = load_jsonl(DATA/"test.jsonl")
if TEST_SHUFFLE:
    random.shuffle(all_test)
test = all_test[:TEST_LIMIT] if TEST_LIMIT and TEST_LIMIT < len(all_test) else all_test

def evaluate(alpha_str):
    q = thresholds[alpha_str]
    covered = 0; help_events = 0; sizes = []
    for ex in test:
        prompt   = client.build_prompt(ex["context"], ex["options"], tmpl)
        probs    = client.get_option_probs(prompt)
        nonconfs = probs_to_nonconfs(probs)
        S        = [k for k,v in nonconfs.items() if v <= q]  # standard inclusion
        if "ABCDE"[ex["correct_index"]] in S: covered += 1
        if len(S) > 1: help_events += 1
        sizes.append(len(S))
    N = len(test)
    return dict(alpha=float(alpha_str),
                coverage=covered/N if N else 0.0,
                avg_set_size=float(np.mean(sizes)) if N else 0.0,
                help_rate=help_events/N if N else 0.0,
                N=N)

for a in sorted(thresholds.keys(), key=float):
    print(evaluate(a))
