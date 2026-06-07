"""Compare v3 argmax predictions against test_llm.py greedy decoded
predictions on BioRED dev to find where the 4.2 pt F1 gap comes from.

For each pair, classify into:
  AGREE       — both predicted the same label
  TESTLLM_NONE_v3_REL  — test_llm said None, v3 picked a rel
  TESTLLM_REL_v3_NONE  — test_llm picked a rel, v3 said None
  REL_MISMATCH — both non-None but different rel
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

V3_PATH = Path("posthoc/v3_results/processed_test_scores.json")
TESTLLM_PATH = Path("posthoc/repro_check/dev_results.txt")

# --- Load v3 predictions ---
v3 = json.load(V3_PATH.open())
v3_pred = {}  # (doc_idx, h, t) -> pred_label
v3_gold = {}
for doc in v3['docs']:
    di = doc['doc_idx']
    for p in doc['pairs']:
        idx = p['logprobs'].index(max(p['logprobs']))
        v3_pred[(di, p['h'], p['t'])] = p['candidates'][idx]
        v3_gold[(di, p['h'], p['t'])] = p['candidates'][p['gold_rel_id']]

# --- Parse test_llm.py output ---
text = TESTLLM_PATH.read_text()
blocks = text.split("____________________________________________________")
# First non-empty block per doc index
ml_pred = {}  # (doc_idx, h, t) -> pred_label  ('None' when test_llm said None)
ml_gold = {}
PAIR_RE = re.compile(r"\{(\d+)\|[^}]+\} -> (\w+) -> \{(\d+)\|")
SECTION_RE = re.compile(r"^(CORRECT|MISSED|INCORRECT) \(\d+\):", re.M)

doc_idx = -1
for block in blocks:
    if "Here is your relation extraction" not in block:
        continue
    doc_idx += 1
    _, _, rest = block.partition("Here is your relation extraction result compared with the ground truth:")
    sections = {"CORRECT": [], "MISSED": [], "INCORRECT": []}
    current = None
    for line in rest.splitlines():
        ms = SECTION_RE.match(line.strip())
        if ms:
            current = ms.group(1)
            continue
        if current and (mp := PAIR_RE.search(line)):
            h, rel, t = int(mp.group(1)), mp.group(2), int(mp.group(3))
            sections[current].append((h, t, rel))
        if line.startswith("F-score"):
            current = None
    for h, t, r in sections["CORRECT"]:
        ml_pred[(doc_idx, h, t)] = r; ml_gold[(doc_idx, h, t)] = r
    for h, t, r in sections["MISSED"]:
        # test_llm said None (missed it), gold is r
        ml_pred[(doc_idx, h, t)] = 'None'; ml_gold[(doc_idx, h, t)] = r
    for h, t, r in sections["INCORRECT"]:
        # test_llm predicted r, gold is None (or different)
        ml_pred[(doc_idx, h, t)] = r
        ml_gold.setdefault((doc_idx, h, t), 'None')

print(f"v3 pairs:      {len(v3_pred)}")
print(f"test_llm pairs:{len(ml_pred)}")
print(f"intersection:  {len(set(v3_pred) & set(ml_pred))}")
print(f"v3 only:       {len(set(v3_pred) - set(ml_pred))}")
print(f"test_llm only: {len(set(ml_pred) - set(v3_pred))}")
print()

# --- Comparison ---
agree = 0
both_none = 0
mismatch = Counter()  # (v3_pred, ml_pred) -> count
gold_mismatch = []
for k in set(v3_pred) & set(ml_pred):
    vp = v3_pred[k]
    mp = ml_pred[k]
    if vp == mp:
        agree += 1
        if vp == 'None':
            both_none += 1
    else:
        mismatch[(vp, mp)] += 1
        if v3_gold.get(k, '?') != ml_gold.get(k, '?'):
            gold_mismatch.append((k, v3_gold[k], ml_gold[k]))

print(f"AGREE: {agree} (both None: {both_none}, both rel: {agree-both_none})")
print(f"MISMATCH: {sum(mismatch.values())}")
print(f"  top mismatch pairs (v3_pred → ml_pred):")
for (vp, mp), n in mismatch.most_common(20):
    print(f"    {vp!r:<25} → {mp!r:<25} : {n}")

print(f"\nGold disagreement between v3 and test_llm: {len(gold_mismatch)}")
if gold_mismatch[:5]:
    print("  Sample gold mismatches (key, v3_gold, test_llm_gold):")
    for k, vg, mg in gold_mismatch[:10]:
        print(f"    {k}: v3={vg!r}  test_llm={mg!r}")
