"""Build a synthetic *_scores.json that emulates real cdr / pharmgkb prior shift.
Used to smoke-test eval_adjusted.py without GPU inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['cdr', 'pharmgkb', 'biored'], required=True)
    ap.add_argument('--n_docs', type=int, default=20)
    ap.add_argument('--pairs_per_doc', type=int, default=40)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--seed', type=int, default=66)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.scenario == 'cdr':
        rel_list = ['CID']
        dataset_name = 'BC5CDR'
        # gold p: 0.65 None, 0.35 CID
        # base model under-predicts: it would say None ~85% of the time
        # i.e. logit bias toward None
        gold_p = np.array([0.65, 0.35])
        logit_bias = np.array([1.5, 0.0])
    elif args.scenario == 'pharmgkb':
        rel_list = ['Association']
        dataset_name = 'PharmGKB'
        # gold p: 0.85 None, 0.15 Association
        # base model over-predicts: would say Association ~80%
        gold_p = np.array([0.85, 0.15])
        logit_bias = np.array([0.0, 2.0])  # bias toward Association
    else:  # biored
        rel_list = ['Association', 'Bind', 'Comparison', 'Conversion',
                    'Cotreatment', 'Drug_Interaction',
                    'Negative_Correlation', 'Positive_Correlation']
        dataset_name = 'BioRED'
        gold_p = np.array([0.70, 0.10, 0.03, 0.01, 0.01, 0.04, 0.005, 0.045, 0.06])
        gold_p = gold_p / gold_p.sum()
        logit_bias = np.zeros(9)

    candidates = ['None'] + list(rel_list)
    n_classes = len(candidates)

    docs = []
    for d in range(args.n_docs):
        pairs = []
        for k in range(args.pairs_per_doc):
            gold = int(rng.choice(n_classes, p=gold_p))
            # Logits: base ~ Normal noise + bias + boost on gold (if non-None)
            logits = rng.normal(scale=1.0, size=n_classes)
            logits -= logit_bias  # bias means: increase the chance to predict that class
            if gold != 0:
                logits[gold] += rng.uniform(0.3, 2.0)
            # Convert to logprobs by softmax
            x = logits - logits.max()
            logp = x - np.log(np.exp(x).sum())
            pairs.append({
                'h': k, 't': (k + 1) % args.pairs_per_doc,
                'h_type': 'Chemical', 't_type': 'Disease',
                'gold_rel_id': gold,
                'logprobs': logp.tolist(),
                'candidates': candidates,
            })
        docs.append({
            'doc_idx': d,
            'dataset_name': dataset_name,
            'rel_list': rel_list,
            'pairs': pairs,
        })

    payload = {
        'source_file': f'<synthetic-{args.scenario}>',
        'variant_dir': '<synthetic>',
        'use_direction': False,
        'n_docs': args.n_docs,
        'n_pairs': args.n_docs * args.pairs_per_doc,
        'docs': docs,
    }
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
