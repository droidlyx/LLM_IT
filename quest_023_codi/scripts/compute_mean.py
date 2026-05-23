# scripts/compute_mean.py
"""Compute global mean over all label templates via base model + CODI student template."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from src.codi_model import load_codi_trainable
from src.codi_inputs import build_student_inputs, pool_positions

LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
OUT_MEAN = "/home/ds/quest_023_codi/artifacts/mean.pt"

os.makedirs(os.path.dirname(OUT_MEAN), exist_ok=True)

print("Loading model + latents (fresh init)...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
model.eval(); backbone.eval()
for p in model.parameters(): p.requires_grad_(False)
for p in latents.parameters(): p.requires_grad_(False)

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
all_templates = [t for l in labels for t in label_dict[l]]
print(f"  {len(all_templates)} templates", flush=True)

embs = []
with torch.inference_mode():
    for i in range(0, len(all_templates), 16):
        chunk = all_templates[i:i+16]
        s_embeds, s_mask, s_pool = build_student_inputs(chunk, tok, model, latents, ids, max_task_len=64)
        out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
        h = pool_positions(out.last_hidden_state.float(), s_pool)
        embs.append(h)
embs = torch.cat(embs, dim=0)
mean = embs.mean(0)
print(f"  mean norm: {mean.norm().item():.2f}", flush=True)
torch.save(mean.cpu(), OUT_MEAN)
print(f"Saved {OUT_MEAN}", flush=True)
