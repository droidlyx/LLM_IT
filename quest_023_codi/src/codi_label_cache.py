"""Encode each label template via [text] [<think>] [k latents] [<answer>] and pool same way as judgments."""
import random
import torch
import torch.nn.functional as F

from .codi_inputs import build_student_inputs, pool_positions


@torch.no_grad()
def encode_label_templates_codi(backbone, tok, model, latents, ids: dict,
                                  label_dict: dict, mean: torch.Tensor,
                                  device: str = "cuda") -> torch.Tensor:
    """Returns tensor of shape (n_labels, max_templates_per_label, D), centered + L2 normalized."""
    labels = list(label_dict.keys())
    max_t = max(len(label_dict[l]) for l in labels)
    embs_per_label = []
    for lbl in labels:
        templates = label_dict[lbl]
        inputs_embeds, attn, pool_idx = build_student_inputs(
            templates, tok, model, latents, ids, max_task_len=64
        )
        out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        h_pool = pool_positions(out.last_hidden_state.float(), pool_idx)  # (T, D)
        centered = h_pool - mean.to(h_pool.dtype).unsqueeze(0)
        normalized = F.normalize(centered, p=2, dim=-1)
        # Pad to max_t by repeating last template
        if normalized.shape[0] < max_t:
            pad = normalized[-1:].expand(max_t - normalized.shape[0], -1)
            normalized = torch.cat([normalized, pad], dim=0)
        embs_per_label.append(normalized)
    return torch.stack(embs_per_label, dim=0)  # (n_labels, max_t, D)


class CODILabelCache:
    """Per-step sampler: returns one template embedding per label."""
    def __init__(self, label_dict, embs_all):
        self.labels = list(label_dict.keys())
        self.embs_all = embs_all  # (n_labels, max_t, D)
        self.n_templates = {i: len(label_dict[l]) for i, l in enumerate(self.labels)}

    def sample_per_label(self, rng: random.Random) -> torch.Tensor:
        rows = []
        for i in range(len(self.labels)):
            t = rng.randrange(self.n_templates[i])
            rows.append(self.embs_all[i, t])
        return torch.stack(rows, dim=0)  # (n_labels, D)
