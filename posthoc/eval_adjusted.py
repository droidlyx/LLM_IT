"""Apply post-hoc methods to a *_scores.json and report F1 deltas.

For each scoring JSON:
  - Compute baseline F1 (argmax)
  - Compute F1 under:
      * LA (tau in {0.5, 1.0, 2.0}) — needs only p_train estimable from gold
      * P2P with oracle p_target (estimated from gold)
      * P2P with self-estimated p_target (no oracle; uses p_eff as proxy)
      * TECP with alpha = 0.10 (calibrated on first half, applied on second)
      * P2P + TECP composition
  - Print before/after table; save per-method per-class breakdown JSON.

Optionally writes a `*_results_<method>.txt` in same format as test_llm.py
output, so the existing analyze_8b_errors.py keeps working downstream.

Usage:
  python posthoc/eval_adjusted.py --scores_json posthoc/results/cdr_scores.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from posthoc_methods import (
    PRF,
    baseline_argmax,
    logit_adjust,
    p2p_adjust,
    tecp_apply,
    tecp_calibrate,
)


def bidirectional_prf(preds: np.ndarray, pairs_meta: list,
                       use_direction: bool, none_idx: int = 0) -> PRF:
    """Compute micro PRF matching test_llm.py semantics.

    For use_direction=False: each per-direction non-None prediction is
    duplicated to (t, h) too. Same for gold.
    """
    pred_set = set()
    gold_set = set()
    for i, m in enumerate(pairs_meta):
        di, h, t = m['doc_idx'], m['h'], m['t']
        p = int(preds[i])
        g = int(m['gold'])
        if p != none_idx:
            pred_set.add((di, h, t, p))
            if not use_direction:
                pred_set.add((di, t, h, p))
        if g != none_idx:
            gold_set.add((di, h, t, g))
            if not use_direction:
                gold_set.add((di, t, h, g))
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    f1 = 2 * pr * rc / max(1e-9, pr + rc)
    return PRF(pr, rc, f1, tp, fp, fn)


def bidirectional_per_class_prf(preds: np.ndarray, pairs_meta: list, candidates: list,
                                  use_direction: bool, none_idx: int = 0) -> dict:
    """Per-class PRF with bidirectional duplication semantics."""
    out = {}
    for c_idx, cls_name in enumerate(candidates):
        if c_idx == none_idx:
            continue
        pred_set = set()
        gold_set = set()
        for i, m in enumerate(pairs_meta):
            di, h, t = m['doc_idx'], m['h'], m['t']
            if int(preds[i]) == c_idx:
                pred_set.add((di, h, t))
                if not use_direction:
                    pred_set.add((di, t, h))
            if int(m['gold']) == c_idx:
                gold_set.add((di, h, t))
                if not use_direction:
                    gold_set.add((di, t, h))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        out[cls_name] = PRF(pr, rc, f1, tp, fp, fn)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_json", required=True,
                   help="path to one *_scores.json from score_pairs.py")
    p.add_argument("--use_direction", action="store_true",
                   help="if False, count (i,j) and (j,i) as the same relation when matching gold")
    p.add_argument("--tecp_alpha", type=float, default=0.10)
    p.add_argument("--report_dir", default="",
                   help="if set, write per-method JSON breakdown + adjusted-prediction *.txt files here")
    p.add_argument("--la_taus", default="0.5,1.0,2.0")
    return p.parse_args()


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return data


def flatten_pairs(data, use_direction=False):
    """Return logits matrix (n_pairs_unique, n_labels), gold array, and pair-meta list.

    For !use_direction, we deduplicate by frozenset({h,t}) keeping the max
    per-label logprob across both directions (since gold and pred are
    symmetric under no-direction setting).
    """
    candidates = None
    # No dedup: keep all per-direction entries. The bidirectional duplication
    # semantics (matching test_llm.py for use_direction=False) are applied at
    # eval-time in the bidirectional_prf functions below.
    pairs_meta = []
    for doc in data['docs']:
        for p in doc['pairs']:
            if candidates is None:
                candidates = p['candidates']
            pairs_meta.append({
                'doc_idx': doc['doc_idx'],
                'h': p['h'], 't': p['t'],
                'h_type': p['h_type'], 't_type': p['t_type'],
                'logprobs': list(p['logprobs']),
                'gold': p['gold_rel_id'],
            })
    logits = np.array([m['logprobs'] for m in pairs_meta], dtype=np.float64)
    gold = np.array([m['gold'] for m in pairs_meta], dtype=np.int64)
    return candidates, logits, gold, pairs_meta


def estimate_oracle_prior(gold: np.ndarray, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    counts = np.bincount(gold, minlength=n_classes).astype(float)
    counts += smoothing
    return counts / counts.sum()


def split_half(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    return idx[:mid], idx[mid:]


def main():
    args = parse_args()
    data = load_scores(args.scores_json)
    print(f"Loaded {args.scores_json}")
    print(f"  n_docs: {data['n_docs']}, n_pairs (raw): {data['n_pairs']}")
    print(f"  variant_dir: {data['variant_dir']}")
    print(f"  use_direction (in scoring): {data['use_direction']}")

    candidates, logits, gold, pairs_meta = flatten_pairs(
        data, use_direction=args.use_direction
    )
    assert candidates is not None, "scores JSON has no pairs"
    n_pairs, n_classes = logits.shape
    print(f"  after dedup: n_pairs = {n_pairs}, n_classes = {n_classes}")
    print(f"  candidates: {candidates}")
    print(f"  gold class distribution: {Counter(gold.tolist())}")

    none_idx = 0
    results = {}

    # ---------- Baseline ----------
    preds_base = baseline_argmax(logits)
    prf_base = bidirectional_prf(preds_base, pairs_meta, args.use_direction, none_idx=none_idx)
    pc_base = bidirectional_per_class_prf(preds_base, pairs_meta, candidates, args.use_direction, none_idx=none_idx)
    print(f"\n[Baseline argmax]              {prf_base}")
    results['baseline'] = {
        'micro': vars(prf_base),
        'per_class': {k: vars(v) for k, v in pc_base.items()},
        'pred_distribution': Counter(preds_base.tolist()),
    }

    # ---------- LA / PAS with multiple tau, using oracle p_train estimated from gold ----------
    p_train_oracle = estimate_oracle_prior(gold, n_classes)
    print(f"\n  p_train (oracle from gold): {np.round(p_train_oracle, 4)}")
    for tau_str in args.la_taus.split(','):
        tau = float(tau_str)
        preds_la, _ = logit_adjust(logits, p_train_oracle, tau=tau)
        prf_la = bidirectional_prf(preds_la, pairs_meta, args.use_direction, none_idx=none_idx)
        pc_la = bidirectional_per_class_prf(preds_la, pairs_meta, candidates, args.use_direction, none_idx=none_idx)
        print(f"[LA  tau={tau:.1f} oracle p_train]   {prf_la}  Δf1={prf_la.f1 - prf_base.f1:+.4f}")
        results[f'LA_tau{tau:.1f}'] = {
            'micro': vars(prf_la), 'tau': tau,
            'per_class': {k: vars(v) for k, v in pc_la.items()},
        }

    # ---------- P2P with oracle p_target ----------
    preds_p2p, p2p_meta = p2p_adjust(logits, p_target=p_train_oracle)
    prf_p2p = bidirectional_prf(preds_p2p, pairs_meta, args.use_direction, none_idx=none_idx)
    pc_p2p = bidirectional_per_class_prf(preds_p2p, pairs_meta, candidates, args.use_direction, none_idx=none_idx)
    print(f"[P2P oracle p_target]          {prf_p2p}  Δf1={prf_p2p.f1 - prf_base.f1:+.4f}")
    print(f"  p_eff (self-estimated): {np.round(p2p_meta['p_eff'], 4)}")
    print(f"  adjustment vector:      {np.round(p2p_meta['adjustment'], 4)}")
    results['P2P_oracle'] = {
        'micro': vars(prf_p2p),
        'p_eff': p2p_meta['p_eff'],
        'p_target': p2p_meta['p_target'],
        'per_class': {k: vars(v) for k, v in pc_p2p.items()},
    }

    # ---------- P2P with uniform p_target (no oracle) ----------
    p_uniform = np.ones(n_classes) / n_classes
    preds_unif, _ = p2p_adjust(logits, p_target=p_uniform)
    prf_unif = bidirectional_prf(preds_unif, pairs_meta, args.use_direction, none_idx=none_idx)
    print(f"[P2P uniform p_target]         {prf_unif}  Δf1={prf_unif.f1 - prf_base.f1:+.4f}")
    results['P2P_uniform'] = {'micro': vars(prf_unif)}

    # ---------- TECP: split-half calibrate, then apply ----------
    if n_pairs >= 50:
        cal_idx, test_idx = split_half(n_pairs, seed=0)
        thr = tecp_calibrate(
            logits[cal_idx], gold[cal_idx],
            none_label_idx=none_idx, alpha=args.tecp_alpha,
        )
        preds_tecp_full, meta_tecp = tecp_apply(logits, thr, none_label_idx=none_idx)
        prf_tecp = bidirectional_prf(preds_tecp_full, pairs_meta, args.use_direction, none_idx=none_idx)
        print(f"[TECP alpha={args.tecp_alpha}, thr={thr:.3f}]  {prf_tecp}  "
              f"abstain={meta_tecp['abstain_rate']:.3f}  Δf1={prf_tecp.f1 - prf_base.f1:+.4f}")
        results['TECP'] = {
            'micro': vars(prf_tecp),
            'threshold': thr,
            'alpha': args.tecp_alpha,
            'abstain_rate': meta_tecp['abstain_rate'],
        }

        # ---------- P2P + TECP composition ----------
        # Re-derive P2P logits, then apply TECP threshold (re-calibrated on adj logits)
        logits_adj = logits + (np.log(p_train_oracle + 1e-9) -
                               np.log(np.asarray(p2p_meta['p_eff']) + 1e-9))[None, :]
        thr_adj = tecp_calibrate(
            logits_adj[cal_idx], gold[cal_idx],
            none_label_idx=none_idx, alpha=args.tecp_alpha,
        )
        preds_combo, meta_combo = tecp_apply(logits_adj, thr_adj, none_label_idx=none_idx)
        prf_combo = bidirectional_prf(preds_combo, pairs_meta, args.use_direction, none_idx=none_idx)
        print(f"[P2P+TECP combo]              {prf_combo}  abstain={meta_combo['abstain_rate']:.3f}  Δf1={prf_combo.f1 - prf_base.f1:+.4f}")
        results['P2P+TECP'] = {
            'micro': vars(prf_combo),
            'threshold': thr_adj,
            'abstain_rate': meta_combo['abstain_rate'],
        }

    # ---------- Best per-class breakdown summary ----------
    print("\nPer-class F1 (baseline → best non-baseline method):")
    method_keys = [k for k in results if k != 'baseline' and 'per_class' in results[k]]
    if method_keys:
        for cls_name in [c for c in candidates if c != 'None']:
            base_f1 = pc_base[cls_name].f1
            best_method, best_f1 = None, base_f1
            for mk in method_keys:
                pc = results[mk]['per_class']
                f1 = pc[cls_name]['f1']
                if f1 > best_f1:
                    best_method, best_f1 = mk, f1
            tag = f" ({best_method})" if best_method else ""
            print(f"  {cls_name:<25} {base_f1:.4f}  ->  {best_f1:.4f}{tag}")

    # ---------- Save report ----------
    if args.report_dir:
        rd = Path(args.report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        scores_name = Path(args.scores_json).stem.replace('_scores', '')
        report_path = rd / f"{scores_name}_adjusted.json"
        # Serialize Counter for JSON
        for k, v in results.items():
            if 'pred_distribution' in v and isinstance(v['pred_distribution'], Counter):
                v['pred_distribution'] = dict(v['pred_distribution'])
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved report: {report_path}")


if __name__ == '__main__':
    main()
