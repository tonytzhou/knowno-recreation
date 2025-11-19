LETTERS = ["A", "B", "C", "D", "E"]

MCQA_TEMPLATE = """You are a careful planner that selects exactly ONE best option.

Context:
{CONTEXT}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

Question: Which option is correct? Answer with just the single letter (A, B, C, D, or E).
"""

def build_mcqa_prompt(context: str, options: list[str]) -> str:
    """
    Build the exact MCQA prompt expected by LLMClient.get_option_probs().
    Ensures the prompt ends with 'Answer: ' so the next token is the letter.
    """
    assert len(options) == 5, "Expected exactly 5 options (A–E)"
    A, B, C, D, E = options
    base = (MCQA_TEMPLATE
            .replace("{CONTEXT}", context)
            .replace("{A}", A).replace("{B}", B)
            .replace("{C}", C).replace("{D}", D)
            .replace("{E}", E))
    return base.rstrip() + "\n\nAnswer: "
