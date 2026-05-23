"""CODI training step: teacher branch + student branch + 4-loss with warmup."""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F

from .codi_inputs import build_student_inputs, build_teacher_inputs, pool_positions


@dataclass
class CODILossConfig:
    k: int = 4
    tau: float = 0.07
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.5
    warmup_steps: int = 200
    ramp_steps: int = 200
    max_task_len: int = 600
    max_reason_len: int = 400


def _warmup_weights(step: int, cfg: CODILossConfig) -> Tuple[float, float]:
    """Linear ramp of β and γ over [warmup_steps, warmup_steps + ramp_steps]."""
    if step < cfg.warmup_steps:
        return 0.0, 0.0
    if step >= cfg.warmup_steps + cfg.ramp_steps:
        return cfg.beta, cfg.gamma
    frac = (step - cfg.warmup_steps) / cfg.ramp_steps
    return cfg.beta * frac, cfg.gamma * frac


def _cosine_logits(judgment: torch.Tensor, label_embs: torch.Tensor, mean: torch.Tensor,
                    tau: float) -> torch.Tensor:
    centered = judgment - mean.to(judgment.dtype).unsqueeze(0)
    rel_norm = F.normalize(centered, p=2, dim=1)
    return (rel_norm @ label_embs.t()) / tau


def codi_step(model, backbone, latents, tok, ids, batch, label_embs: torch.Tensor,
               mean: torch.Tensor, cfg: CODILossConfig, step: int):
    """One training step with teacher + student forwards and 4-loss."""
    device = next(model.parameters()).device
    label_embs = label_embs.to(device=device, dtype=torch.float32)
    mean = mean.to(device=device, dtype=torch.float32)
    gold = batch["label_idx"].to(device)

    # ---- Student forward ----
    s_embeds, s_mask, s_pool = build_student_inputs(
        batch["task_text"], tok, model, latents, ids, max_task_len=cfg.max_task_len
    )
    s_out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
    s_judgment = pool_positions(s_out.last_hidden_state.float(), s_pool)  # (B, D)

    # ---- Teacher forward ----
    t_embeds, t_mask, t_pool = build_teacher_inputs(
        batch["task_text"], batch["reasoning"], tok, model, ids,
        k=cfg.k, max_task_len=cfg.max_task_len, max_reason_len=cfg.max_reason_len
    )
    t_out = backbone(inputs_embeds=t_embeds, attention_mask=t_mask, use_cache=False)
    t_judgment = pool_positions(t_out.last_hidden_state.float(), t_pool)

    # ---- Logits ----
    logits_S = _cosine_logits(s_judgment, label_embs, mean, cfg.tau)
    logits_T = _cosine_logits(t_judgment, label_embs, mean, cfg.tau)

    # ---- Losses ----
    L_cls_S = F.cross_entropy(logits_S, gold)
    L_cls_T = F.cross_entropy(logits_T, gold)

    # Align in centered+normalized vec space
    s_centered_norm = F.normalize(s_judgment - mean.unsqueeze(0), p=2, dim=1)
    with torch.no_grad():
        t_centered_norm = F.normalize(t_judgment - mean.unsqueeze(0), p=2, dim=1)
    L_align = F.mse_loss(s_centered_norm, t_centered_norm)

    with torch.no_grad():
        p_T = F.softmax(logits_T.detach(), dim=-1)
    log_p_S = F.log_softmax(logits_S, dim=-1)
    L_distill = F.kl_div(log_p_S, p_T, reduction="batchmean")

    beta_eff, gamma_eff = _warmup_weights(step, cfg)
    total = L_cls_S + cfg.alpha * L_cls_T + beta_eff * L_align + gamma_eff * L_distill

    with torch.no_grad():
        acc_S = (logits_S.argmax(-1) == gold).float().mean().item()
        acc_T = (logits_T.argmax(-1) == gold).float().mean().item()
        align_gap = (s_centered_norm - t_centered_norm).norm(dim=-1).mean().item()

    log = {
        "L_cls_S": float(L_cls_S.detach()),
        "L_cls_T": float(L_cls_T.detach()),
        "L_align": float(L_align.detach()),
        "L_distill": float(L_distill.detach()),
        "total": float(total.detach()),
        "acc_S": acc_S, "acc_T": acc_T,
        "align_gap": align_gap,
        "beta_eff": beta_eff, "gamma_eff": gamma_eff,
    }
    return total, log
