"""Quick test: re-evaluate v3 BioRED dev WITHOUT deduplicating bidirectional pairs.
Mimics test_llm.py's use_direction=False bidirectional duplication."""
import json
import sys
import numpy as np
from collections import Counter
from pathlib import Path

sys.path.insert(0, 'posthoc')
from posthoc_methods import baseline_argmax, micro_prf

data = json.load(open("posthoc/v3_results/processed_test_scores.json"))
print(f"v3 dev: n_docs={data['n_docs']}, n_pairs (per-direction)={data['n_pairs']}")

# Build per-direction predictions
records = []
for doc in data['docs']:
    for p in doc['pairs']:
        records.append({
            'doc_idx': doc['doc_idx'],
            'h': p['h'], 't': p['t'],
            'logprobs': p['logprobs'],
            'gold': p['gold_rel_id'],
            'candidates': p['candidates'],
        })

print(f"records: {len(records)}")

# Method 1: original eval with dedup (frozenset)
seen = {}
for r in records:
    key = (r['doc_idx'], min(r['h'], r['t']), max(r['h'], r['t']))
    if key not in seen:
        seen[key] = {'logprobs': list(r['logprobs']), 'gold': r['gold'], 'cands': r['candidates']}
    else:
        seen[key]['logprobs'] = [max(a, b) for a, b in zip(seen[key]['logprobs'], r['logprobs'])]
        if r['gold'] != 0:
            seen[key]['gold'] = r['gold']
logits1 = np.array([s['logprobs'] for s in seen.values()])
gold1 = np.array([s['gold'] for s in seen.values()])
preds1 = baseline_argmax(logits1)
prf1 = micro_prf(preds1, gold1, none_idx=0)
print(f"\nMethod 1 (dedup, max-merge): {prf1}")

# Method 2: keep both directions separate (matches test_llm bidirectional)
logits2 = np.array([r['logprobs'] for r in records])
gold2 = np.array([r['gold'] for r in records])
preds2 = baseline_argmax(logits2)
prf2 = micro_prf(preds2, gold2, none_idx=0)
print(f"Method 2 (no dedup, per direction): {prf2}")

# Method 3: test_llm.py exact — each direction's prediction is duplicated to both directions
# (i.e., if (h,t) predicted rel, add (h,t,rel) AND (t,h,rel))
pred_set = set()
gold_set = set()
for r in records:
    di = r['doc_idx']
    h, t = r['h'], r['t']
    pred_idx = int(np.argmax(r['logprobs']))
    if pred_idx != 0:  # non-None prediction
        pred_set.add((di, h, t, pred_idx))
        pred_set.add((di, t, h, pred_idx))  # duplicate bidirectional
    if r['gold'] != 0:
        gold_set.add((di, h, t, r['gold']))
        gold_set.add((di, t, h, r['gold']))
tp = len(pred_set & gold_set)
fp = len(pred_set - gold_set)
fn = len(gold_set - pred_set)
p_ = tp / max(1, tp + fp)
r_ = tp / max(1, tp + fn)
f1 = 2*p_*r_ / max(1e-9, p_ + r_)
print(f"Method 3 (test_llm.py exact bidirectional): P={p_:.4f} R={r_:.4f} F1={f1:.4f} tp={tp} fp={fp} fn={fn}")

print(f"\ntest_llm.py reproduced F1 (just-rerun): 0.6478")
