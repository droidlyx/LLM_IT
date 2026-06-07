"""CPU-only sanity: verify all BioRED + OOD candidates have distinct first tokens.
Run: python posthoc/_check_first_tokens.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

MODEL_PATH = "/root/gpufree-data/Qwen3-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

CANDIDATE_SETS = {
    "BioRED": ["None", "Association", "Bind", "Comparison", "Conversion",
               "Cotreatment", "Drug_Interaction", "Negative_Correlation",
               "Positive_Correlation"],
    "cdr": ["None", "CID"],
    "disgenet": ["None", "Association"],
    "pharmgkb": ["None", "Association"],
}

ANSWER_PREFIX = "1. "
prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
print(f"Answer prefix tokens: {prefix_ids} = {tokenizer.convert_ids_to_tokens(prefix_ids)}")
print()

for name, candidates in CANDIDATE_SETS.items():
    print(f"--- {name} ---")
    seen_tokens = {}
    for label in candidates:
        full_ids = tokenizer.encode(ANSWER_PREFIX + label, add_special_tokens=False)
        i = 0
        while i < len(prefix_ids) and i < len(full_ids) and prefix_ids[i] == full_ids[i]:
            i += 1
        new_tokens = full_ids[i:]
        first_tok = new_tokens[0] if new_tokens else -1
        tok_str = tokenizer.decode([first_tok]) if first_tok >= 0 else "<empty>"
        collision = ""
        if first_tok in seen_tokens:
            collision = f"  ⚠ COLLISION with '{seen_tokens[first_tok]}'"
        seen_tokens[first_tok] = label
        print(f"  {label:<25} first_tok={first_tok} ({tok_str!r}), full new tokens: {new_tokens}{collision}")
    print()
