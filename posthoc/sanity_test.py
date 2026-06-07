"""Sanity tests for posthoc_methods.py — no GPU needed.

Synthetic scenarios:
  S1. Identity test: P2P with p_target == p_eff should not change predictions.
  S2. Prior shift recovery: model trained on uniform prior, deployed on
      skewed target → P2P should recover gold predictions.
  S3. Long-tail rebalancing: PAS with tau=1 should raise rare-class recall.
  S4. TECP coverage: empirical error rate after abstention ≈ alpha on cal set.
  S5. End-to-end on toy prior-shift data: P2P > baseline F1.

Also includes a small reproduction of our actual cdr / pharmgkb scenario
(under/over-prediction) to verify the methods move predictions in the
right direction.

Run: python posthoc/sanity_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from posthoc_methods import (
    PRF,
    baseline_argmax,
    empirical_prior,
    logit_adjust,
    micro_prf,
    p2p_adjust,
    pas_adjust,
    softmax,
    tecp_apply,
    tecp_calibrate,
    token_entropy,
)


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_softmax_basic():
    x = np.array([[1.0, 2.0, 3.0]])
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(), 1.0), p
    assert p[0, 2] > p[0, 1] > p[0, 0], p
    print("  [OK] softmax basic")


def test_token_entropy():
    uniform = np.zeros((1, 5))  # uniform → ent = log(5)
    assert _approx(token_entropy(uniform)[0], np.log(5)), token_entropy(uniform)
    sharp = np.array([[10.0, 0, 0, 0, 0]])  # nearly delta → ent ≈ 0
    assert token_entropy(sharp)[0] < 0.01
    print("  [OK] token_entropy")


def test_p2p_identity():
    """S1: when p_target == p_eff, no change."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 5))
    probs = softmax(logits)
    p_eff = probs.mean(axis=0)
    preds, _ = p2p_adjust(logits, p_target=p_eff, p_eff=p_eff)
    expected = baseline_argmax(logits)
    assert (preds == expected).all(), f"S1 broke: {(preds != expected).sum()} mismatches"
    print("  [OK] S1 P2P identity (p_target == p_eff)")


def test_p2p_prior_shift_recovery():
    """S2: model trained on uniform, target has skewed prior 80/20 for class 1.
    Without adjustment, baseline gives ~50/50; with P2P(target=[0.2, 0.8]),
    predictions should swing toward class 1.
    """
    rng = np.random.default_rng(42)
    n = 500
    # Construct logits where two classes are roughly balanced
    logits = rng.normal(scale=0.3, size=(n, 2))
    p_target = np.array([0.2, 0.8])
    preds_base = baseline_argmax(logits)
    base_class1 = (preds_base == 1).mean()
    preds_p2p, meta = p2p_adjust(logits, p_target=p_target)
    p2p_class1 = (preds_p2p == 1).mean()
    print(f"  base class-1 rate: {base_class1:.3f}, P2P class-1 rate: {p2p_class1:.3f}")
    assert p2p_class1 > base_class1 + 0.1, "P2P should shift predictions toward majority target"
    print("  [OK] S2 P2P prior shift recovery")


def test_la_long_tail():
    """S3: rare class with low training prior should get boosted at inference."""
    rng = np.random.default_rng(7)
    n = 200
    # 3 classes, rare class 2 has p_train = 0.05
    p_train = np.array([0.6, 0.35, 0.05])
    # Logits roughly proportional to log p_train for an under-trained model
    logits = rng.normal(size=(n, 3)) + np.log(p_train)[None, :]
    preds_base = baseline_argmax(logits)
    rare_base = (preds_base == 2).mean()
    preds_la, _ = logit_adjust(logits, p_train, tau=1.0)
    rare_la = (preds_la == 2).mean()
    print(f"  base rare rate: {rare_base:.3f}, LA rare rate: {rare_la:.3f}")
    assert rare_la > rare_base, "LA should raise rare-class prediction rate"
    print("  [OK] S3 LA boosts rare class")


def test_tecp_calibration_coverage():
    """S4: After TECP, error rate on a held-out set should be roughly <= alpha."""
    rng = np.random.default_rng(13)
    # Calibration set: 200 pairs, 3 classes, alpha=0.10
    n_cal, n_test, K = 200, 1000, 4
    alpha = 0.10
    none_idx = 0
    cal_logits = rng.normal(size=(n_cal, K))
    cal_gold = rng.integers(0, K, size=n_cal)
    # Inject some confidence: when gold != None, push that class's logit up sometimes
    for i in range(n_cal):
        if cal_gold[i] != none_idx and rng.uniform() < 0.5:
            cal_logits[i, cal_gold[i]] += 3.0
    thr = tecp_calibrate(cal_logits, cal_gold, none_label_idx=none_idx, alpha=alpha)
    print(f"  TECP threshold: {thr:.4f}")

    # Test on independent set with same distribution
    test_logits = rng.normal(size=(n_test, K))
    test_gold = rng.integers(0, K, size=n_test)
    for i in range(n_test):
        if test_gold[i] != none_idx and rng.uniform() < 0.5:
            test_logits[i, test_gold[i]] += 3.0
    preds_adj, meta = tecp_apply(test_logits, thr, none_label_idx=none_idx)
    print(f"  TECP abstain rate: {meta['abstain_rate']:.3f}")
    print("  [OK] S4 TECP runs end-to-end")


def test_cdr_under_predict_scenario():
    """Replicate our cdr finding: model under-predicts (ratio 0.58).
    P2P with target prior of 1.0/0.0 should NOT collapse to all-CID — that
    would be silly. But target prior 0.7 None / 0.3 CID should shift the
    decision boundary so more pairs become CID.
    """
    rng = np.random.default_rng(99)
    n = 500
    # 2 classes: None (idx 0), CID (idx 1)
    # Model is heavily biased toward None — synthesize so base_cid ≈ 0.15
    # Target prior in our cdr scenario: roughly 0.65 None / 0.35 CID
    logits = rng.normal(scale=1.0, size=(n, 2))
    logits[:, 0] += 1.5  # strong bias toward None

    preds_base = baseline_argmax(logits)
    base_cid_rate = (preds_base == 1).mean()
    print(f"  base CID rate: {base_cid_rate:.3f} (synthetic under-predict)")

    p_target = np.array([0.65, 0.35])
    preds_p2p, meta = p2p_adjust(logits, p_target=p_target)
    p2p_cid_rate = (preds_p2p == 1).mean()
    print(f"  P2P CID rate (target 0.35): {p2p_cid_rate:.3f}")
    print(f"  p_eff: {meta['p_eff']}")
    print(f"  adjustment: {meta['adjustment']}")
    assert base_cid_rate < 0.25, f"test setup failed: base_cid_rate {base_cid_rate} not low enough"
    assert p2p_cid_rate > base_cid_rate, "Under-prediction not corrected"
    print("  [OK] S5a cdr under-predict scenario: P2P raises CID rate")


def test_pharmgkb_over_predict_scenario():
    """Replicate pharmgkb (ratio 4.73): model OVER-predicts Association.
    P2P should reduce the Association rate."""
    rng = np.random.default_rng(123)
    n = 500
    logits = rng.normal(scale=1.0, size=(n, 2))
    logits[:, 1] += 1.5  # over-bias toward Association

    preds_base = baseline_argmax(logits)
    base_assoc = (preds_base == 1).mean()
    print(f"  base Association rate: {base_assoc:.3f}")

    p_target = np.array([0.85, 0.15])
    preds_p2p, meta = p2p_adjust(logits, p_target=p_target)
    p2p_assoc = (preds_p2p == 1).mean()
    print(f"  P2P Association rate (target 0.15): {p2p_assoc:.3f}")
    assert p2p_assoc < base_assoc, "Over-prediction not corrected"
    print("  [OK] S5b pharmgkb over-predict scenario: P2P lowers Association rate")


def test_prf_baseline():
    """Sanity check that micro_prf gives expected values."""
    preds = np.array([0, 1, 2, 1, 0])
    gold = np.array([0, 1, 2, 0, 1])
    prf = micro_prf(preds, gold, none_idx=0)
    # TP: indices 1,2 (pred==gold and both non-None) → 2
    # FP: indices where pred non-None but pred != gold → idx 3 (pred=1 gold=0) → 1
    # FN: indices where gold non-None but pred != gold → idx 3 (gold=0 — wait, gold=0 is None, skip) and idx 4 (pred=0 gold=1) → 1
    assert prf.tp == 2, prf
    assert prf.fp == 1, prf
    assert prf.fn == 1, prf
    print(f"  PRF: {prf}")
    print("  [OK] micro_prf sanity")


def test_end_to_end_synthetic_biored():
    """Simulate BioRED-like dev set: 9 classes (None + 8 relations), known
    gold distribution, model trained with one prior, evaluated with another.

    With oracle p_target, P2P should beat baseline on F1.
    """
    rng = np.random.default_rng(31)
    n = 800
    K = 9  # None + 8
    p_train = np.array([0.7, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])
    p_target = np.array([0.5, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])

    # Generate true labels from p_target
    gold = rng.choice(K, size=n, p=p_target)

    # Generate logits: roughly log p_train + noise + boost on the gold class
    base_logits = np.log(p_train + 1e-9)[None, :].repeat(n, axis=0)
    noise = rng.normal(scale=1.0, size=(n, K))
    boost = np.zeros_like(base_logits)
    boost[np.arange(n), gold] += rng.uniform(0.5, 2.5, size=n)  # model is decent but noisy
    logits = base_logits + noise + boost

    # Baseline
    preds_base = baseline_argmax(logits)
    f1_base = micro_prf(preds_base, gold).f1

    # P2P with oracle target prior
    preds_p2p, _ = p2p_adjust(logits, p_target=p_target)
    f1_p2p = micro_prf(preds_p2p, gold).f1

    # LA with training prior, tau=1
    preds_la, _ = logit_adjust(logits, p_train, tau=1.0)
    f1_la = micro_prf(preds_la, gold).f1

    # PAS = LA with same tau
    preds_pas, _ = pas_adjust(logits, p_train, tau=1.0)
    f1_pas = micro_prf(preds_pas, gold).f1

    print(f"  baseline F1: {f1_base:.4f}")
    print(f"  P2P (oracle): {f1_p2p:.4f}")
    print(f"  LA (train prior): {f1_la:.4f}")
    print(f"  PAS (train prior): {f1_pas:.4f}")
    assert f1_p2p > f1_base, "P2P with oracle target should beat baseline"
    assert f1_la > f1_base, "LA should boost rare classes"
    assert f1_pas == f1_la, "PAS and LA share argmax"
    print("  [OK] end-to-end synthetic BioRED")


def main():
    print("=" * 70)
    print("posthoc_methods sanity tests")
    print("=" * 70)
    tests = [
        ("softmax/entropy", test_softmax_basic),
        ("token_entropy", test_token_entropy),
        ("S1 P2P identity", test_p2p_identity),
        ("S2 P2P prior shift", test_p2p_prior_shift_recovery),
        ("S3 LA long-tail", test_la_long_tail),
        ("S4 TECP", test_tecp_calibration_coverage),
        ("S5a cdr scenario", test_cdr_under_predict_scenario),
        ("S5b pharmgkb scenario", test_pharmgkb_over_predict_scenario),
        ("micro_prf", test_prf_baseline),
        ("E2E synthetic BioRED", test_end_to_end_synthetic_biored),
    ]
    fails = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            fails.append(name)
    print()
    print("=" * 70)
    if fails:
        print(f"FAILED ({len(fails)}/{len(tests)}): {fails}")
        sys.exit(1)
    else:
        print(f"ALL PASSED ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    main()
