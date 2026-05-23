"""Build inputs_embeds + attention_mask + pool position indices for teacher/student branches.

Both branches share the layout: [task_input] [<think>] [...] [<answer>].
Student: middle = k learnable latents (inserted as embeddings, not tokens).
Teacher: middle = reasoning text tokens.

Padding is LEFT (consistent with quest 022 convention).
Pool positions are the last k+1 token positions of each row (the latents/last-k-reason + <answer>).
"""
from typing import List, Tuple
import torch


def _left_pad_ids(rows: List[List[int]], pad_id: int):
    L = max(len(r) for r in rows)
    out_ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    out_mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        out_ids[i, L - len(r):] = torch.tensor(r, dtype=torch.long)
        out_mask[i, L - len(r):] = 1
    return out_ids, out_mask


def _truncate_left(ids: List[int], max_len: int):
    return ids[-max_len:] if len(ids) > max_len else ids


def build_student_inputs(task_texts: List[str], tok, model, latents, ids: dict,
                          max_task_len: int = 600):
    """
    Output:
      inputs_embeds: (B, L, D)  — task_emb || think_emb || k latent_emb || answer_emb
      attention_mask: (B, L)
      pool_idx: (B, k+1) — absolute positions of the k latents + <answer>
    """
    device = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    k = latents.k

    # Tokenize tasks
    task_ids_list = []
    for t in task_texts:
        tids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        task_ids_list.append(_truncate_left(tids, max_task_len))
    pad_id = tok.pad_token_id
    task_ids, task_mask = _left_pad_ids(task_ids_list, pad_id)
    task_ids = task_ids.to(device); task_mask = task_mask.to(device)

    # Embed task
    task_emb = emb_layer(task_ids)  # (B, L_task, D)
    B, L_task, D = task_emb.shape

    think_id = ids["think"]; answer_id = ids["answer"]
    think_emb = emb_layer(torch.tensor([think_id], device=device)).expand(B, 1, D)
    answer_emb = emb_layer(torch.tensor([answer_id], device=device)).expand(B, 1, D)
    latent_emb = latents(B).to(task_emb.dtype)  # (B, k, D)

    inputs_embeds = torch.cat([task_emb, think_emb, latent_emb, answer_emb], dim=1)

    suffix_mask = torch.ones(B, k + 2, dtype=torch.long, device=device)
    attention_mask = torch.cat([task_mask, suffix_mask], dim=1)

    L = inputs_embeds.shape[1]
    # Pool positions: last k+1 absolute indices (latents + <answer>)
    # Suffix is identical length across rows, sitting at tail.
    pool_idx = torch.arange(L - (k + 1), L, device=device).unsqueeze(0).expand(B, -1)

    return inputs_embeds, attention_mask, pool_idx


def build_teacher_inputs(task_texts: List[str], reasonings: List[str], tok, model,
                          ids: dict, k: int = 4, max_task_len: int = 600,
                          max_reason_len: int = 400):
    """
    Output: same shape contract as student. Middle = reasoning tokens.
    Pool positions = last (k+1) positions = last k reasoning tokens + <answer>.
    """
    device = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    pad_id = tok.pad_token_id
    think_id = ids["think"]; answer_id = ids["answer"]

    rows = []
    for t, r in zip(task_texts, reasonings):
        tids = _truncate_left(tok(t, add_special_tokens=False)["input_ids"], max_task_len)
        rids = tok(r, add_special_tokens=False)["input_ids"][:max_reason_len]
        # Enforce reasoning has at least k tokens (pad with answer_id if too short)
        if len(rids) < k:
            rids = rids + [answer_id] * (k - len(rids))
        rows.append(tids + [think_id] + rids + [answer_id])

    full_ids, full_mask = _left_pad_ids(rows, pad_id)
    full_ids = full_ids.to(device); full_mask = full_mask.to(device)
    inputs_embeds = emb_layer(full_ids)

    B, L = full_ids.shape
    # Last position is <answer>. The k tokens before it are the last k reasoning tokens.
    pool_idx = torch.arange(L - (k + 1), L, device=device).unsqueeze(0).expand(B, -1)

    return inputs_embeds, full_mask, pool_idx


def pool_positions(hidden_states: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
    """hidden_states (B, L, D); pool_idx (B, P). Return (B, D) mean-pooled."""
    B, P = pool_idx.shape
    D = hidden_states.shape[-1]
    gather_idx = pool_idx.unsqueeze(-1).expand(-1, -1, D)
    picked = torch.gather(hidden_states, dim=1, index=gather_idx)  # (B, P, D)
    return picked.mean(dim=1)
