"""Post-hoc calibration methods for LLM-IFT relation predictions.

Given per-(pair, label) logprob arrays from score_pairs.py, we implement:

  - P2P: Prior2Posterior logit adjustment (arXiv 2412.16540)
         logit_adjusted[c] = logit[c] - log P_eff[c] + log P_target[c]
         where P_eff is estimated from unlabeled predictions, P_target is the
         (oracle or estimated) target distribution.

  - PAS: Prevalence-Adjusted Softmax (arXiv 2507.06867)
         softmax_adjusted[c] = softmax(logit[c] - log P_train[c])
         The training-prior subtraction is the long-tail "Logit Adjustment"
         loss-form analogue at inference. Targets macro-coverage.

  - LA (Logit Adjustment, Menon et al. 2021): same as PAS without prevalence.
         logit_adj[c] = logit[c] - tau * log P_train[c]
         tau in [0.5, 2.0], standard long-tail trick.

  - TECP: Token-Entropy Conformal Prediction (arXiv 2509.00461)
          For each pair, compute token entropy on the model's chosen label.
          Set a CP threshold s.t. coverage >= 1 - alpha on a calibration set;
          abstain (predict "None") on pairs above the threshold.

  - SET: Greedy argmax (no adjustment), the baseline.

All methods take the same input format:

  logits: np.ndarray shape (n_pairs, n_labels)
  gold_labels: np.ndarray shape (n_pairs,)  # 0 = None, 1..N-1 = rel
  candidates: list[str]  # ["None", "Association", ...]

and return:

  preds: np.ndarray shape (n_pairs,)  # adjusted predictions
  meta: dict with diagnostic info

Math is unit-tested in sanity_test.py (no GPU needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _to_array(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def empirical_prior(preds: np.ndarray, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    """Laplace-smoothed empirical class frequency from predictions or labels."""
    counts = np.bincount(preds, minlength=n_classes).astype(float)
    counts += smoothing
    return counts / counts.sum()


# ----------------------------------------------------------------------------
# Baseline: argmax
# ----------------------------------------------------------------------------

def baseline_argmax(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=-1)


# ----------------------------------------------------------------------------
# P2P: Prior2Posterior
# ----------------------------------------------------------------------------

def p2p_adjust(
    logits: np.ndarray,
    p_target: np.ndarray,
    p_eff: Optional[np.ndarray] = None,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """Prior2Posterior: subtract estimated effective prior, add target prior.

    p_eff (if None) is estimated as the marginal of softmax(logits), i.e.,
    the model's own implicit prior. This is the P2P "effective prior"
    (arXiv:2412.16540 eqs 5-6).

    Returns (preds, meta) where meta contains the priors used.
    """
    logits = _to_array(logits).astype(np.float64)
    n_classes = logits.shape[1]
    p_target_n = _to_array(p_target).astype(np.float64)
    assert p_target_n.shape == (n_classes,), f"p_target shape {p_target_n.shape} != ({n_classes},)"
    p_target_n = p_target_n / p_target_n.sum()

    if p_eff is None:
        p_eff_n = softmax(logits, axis=-1).mean(axis=0)
    else:
        p_eff_n = _to_array(p_eff).astype(np.float64)
        p_eff_n = p_eff_n / p_eff_n.sum()

    adjustment = np.log(p_target_n + eps) - np.log(p_eff_n + eps)
    logits_adj = logits + adjustment[None, :]
    preds = logits_adj.argmax(axis=-1)

    return preds, {
        "method": "P2P",
        "p_eff": p_eff_n.tolist(),
        "p_target": p_target_n.tolist(),
        "adjustment": adjustment.tolist(),
    }


# ----------------------------------------------------------------------------
# Logit Adjustment (Menon et al. 2021) / PAS
# ----------------------------------------------------------------------------

def logit_adjust(
    logits: np.ndarray,
    p_train: np.ndarray,
    tau: float = 1.0,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """LA: subtract tau * log P_train from logits at inference time.

    tau = 1.0 is the standard form. tau in [0.5, 2.0] is the search range
    used in long-tail papers.
    """
    logits = _to_array(logits).astype(np.float64)
    p_train = _to_array(p_train).astype(np.float64)
    p_train = p_train / p_train.sum()

    adjustment = -tau * np.log(p_train + eps)
    logits_adj = logits + adjustment[None, :]
    preds = logits_adj.argmax(axis=-1)
    return preds, {
        "method": "LA",
        "tau": tau,
        "p_train": p_train.tolist(),
    }


def pas_adjust(
    logits: np.ndarray,
    p_train: np.ndarray,
    tau: float = 1.0,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """PAS = LA's softmax-form. Identical argmax behavior with same tau.
    Provided separately so callers can label results clearly."""
    preds, meta = logit_adjust(logits, p_train, tau=tau, eps=eps)
    meta["method"] = "PAS"
    return preds, meta


# ----------------------------------------------------------------------------
# TECP: Token-Entropy Conformal Prediction
# ----------------------------------------------------------------------------

def token_entropy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy of softmax(logits) in nats."""
    p = softmax(logits, axis=axis)
    p_safe = np.clip(p, 1e-12, 1.0)
    return -(p_safe * np.log(p_safe)).sum(axis=axis)


def tecp_calibrate(
    cal_logits: np.ndarray,
    cal_gold: np.ndarray,
    none_label_idx: int = 0,
    alpha: float = 0.10,
) -> float:
    """Calibrate the entropy threshold on a held-out set so that empirical
    error rate <= alpha.

    Returns: threshold tau such that "abstain if entropy > tau" achieves the
    target coverage on the calibration set.

    For RE, "error" = predicted relation != gold relation AND gold != None.
    We want to abstain (predict None) on high-entropy errors but keep
    confident correct predictions.
    """
    cal_logits = _to_array(cal_logits)
    cal_gold = _to_array(cal_gold)
    n = len(cal_gold)
    if n == 0:
        return float("inf")

    entropies = token_entropy(cal_logits)
    preds = cal_logits.argmax(axis=-1)

    # A "bad commit" is any non-None prediction that disagrees with gold:
    # this includes false positives (gold=None) AND wrong-class predictions.
    # These are the predictions where abstaining (=predict None) would help.
    bad_commit = (preds != none_label_idx) & (preds != cal_gold)
    if bad_commit.sum() == 0:
        # No errors to calibrate against — never abstain.
        return float("inf")

    # Sort bad-commit entropies ascending; take the entropy below which
    # alpha fraction of bad commits sit. Above the threshold, abstain.
    bad_ent = np.sort(entropies[bad_commit])
    n_bad = len(bad_ent)
    # We want to catch ≥ (1 - alpha) fraction of bad commits via abstention.
    # Index at the (alpha)-th percentile of bad entropies.
    idx = max(0, int(np.floor(alpha * n_bad)))
    threshold = float(bad_ent[idx])
    return threshold


def tecp_apply(
    logits: np.ndarray,
    threshold: float,
    none_label_idx: int = 0,
) -> tuple[np.ndarray, dict]:
    """Apply TECP: predict argmax unless entropy > threshold, in which case
    abstain (predict None).
    """
    logits = _to_array(logits)
    entropies = token_entropy(logits)
    preds = logits.argmax(axis=-1)
    abstained = entropies > threshold
    preds_adj = np.where(abstained, none_label_idx, preds)
    return preds_adj, {
        "method": "TECP",
        "threshold": threshold,
        "n_abstained": int(abstained.sum()),
        "n_total": len(preds),
        "abstain_rate": float(abstained.mean()),
    }


# ----------------------------------------------------------------------------
# Composition: P2P + TECP, etc.
# ----------------------------------------------------------------------------

def compose_p2p_tecp(
    logits: np.ndarray,
    p_target: np.ndarray,
    tecp_threshold: float,
    p_eff: Optional[np.ndarray] = None,
    none_label_idx: int = 0,
) -> tuple[np.ndarray, dict]:
    """First P2P-adjust logits, then TECP-abstain. P2P moves the distribution
    closer to target prior, TECP catches remaining high-entropy errors."""
    logits = _to_array(logits).astype(np.float64)
    p_target_n = _to_array(p_target).astype(np.float64)
    p_target_n = p_target_n / p_target_n.sum()
    if p_eff is None:
        p_eff_n = softmax(logits, axis=-1).mean(axis=0)
    else:
        p_eff_n = _to_array(p_eff).astype(np.float64)
        p_eff_n = p_eff_n / p_eff_n.sum()
    adjustment = np.log(p_target_n + 1e-9) - np.log(p_eff_n + 1e-9)
    logits_adj = logits + adjustment[None, :]
    preds, tecp_meta = tecp_apply(logits_adj, tecp_threshold, none_label_idx=none_label_idx)
    tecp_meta["method"] = "P2P+TECP"
    tecp_meta["p_eff"] = p_eff_n.tolist()
    tecp_meta["p_target"] = p_target_n.tolist()
    return preds, tecp_meta


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------

@dataclass
class PRF:
    p: float
    r: float
    f1: float
    tp: int
    fp: int
    fn: int

    def __repr__(self) -> str:
        return f"PRF(p={self.p:.4f} r={self.r:.4f} f1={self.f1:.4f} tp={self.tp} fp={self.fp} fn={self.fn})"


def micro_prf(preds: np.ndarray, gold: np.ndarray, none_idx: int = 0) -> PRF:
    """Micro-PRF over non-None labels (treats None as no-relation).
    A prediction matches gold if BOTH non-None AND label matches.
    """
    preds = _to_array(preds)
    gold = _to_array(gold)
    pred_pos = preds != none_idx
    gold_pos = gold != none_idx
    tp = int(((preds == gold) & pred_pos & gold_pos).sum())
    fp = int((pred_pos & ~(preds == gold)).sum())
    fn = int((gold_pos & ~(preds == gold)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return PRF(p, r, f1, tp, fp, fn)


def per_class_prf(preds: np.ndarray, gold: np.ndarray, candidates: list[str],
                  none_idx: int = 0) -> dict[str, PRF]:
    """Per-class P/R/F1 (excluding None)."""
    preds = _to_array(preds)
    gold = _to_array(gold)
    out = {}
    for c, name in enumerate(candidates):
        if c == none_idx:
            continue
        p_mask = preds == c
        g_mask = gold == c
        tp = int((p_mask & g_mask).sum())
        fp = int((p_mask & ~g_mask).sum())
        fn = int((~p_mask & g_mask).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        out[name] = PRF(p, r, f1, tp, fp, fn)
    return out
