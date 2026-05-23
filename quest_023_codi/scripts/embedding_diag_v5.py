"""Re-run embedding crowding diagnostic on v5 adapter judgment vectors.

Compares against v4's mean inter-label cosine (~0.40 training, ~0.94 worst pair).
Acceptance gate: opposite-direction pair cosine < 0.7.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from scripts.eval_codi import load_codi_eval
from src.codi_label_cache import encode_label_templates_codi

OUT_DIR = "/home/ds/quest_023_codi/artifacts/smoke_v5"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = f"{OUT_DIR}/mean.pt"

model, backbone, tok, latents, ids = load_codi_eval(
    f"{OUT_DIR}/lora_adapter", f"{OUT_DIR}/tokenizer",
    f"{OUT_DIR}/latents.pt", f"{OUT_DIR}/special_ids.json", k=4
)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
label_embs = F.normalize(embs_all.mean(dim=1), p=2, dim=-1)

sim = label_embs @ label_embs.t()
n = sim.shape[0]
off = sim - torch.eye(n, device=sim.device)
mean_off = off.sum().item() / (n * (n - 1))
max_off = off.max().item()

print(f"Training labels (n={n}) inter-cosine:")
print(f"  mean: {mean_off:+.3f}  (v4 baseline: +0.401)")
print(f"  max:  {max_off:+.3f}  (v4 baseline: +0.936)")

pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((sim[i, j].item(), labels[i], labels[j]))
pairs.sort(reverse=True)
print("Top-5 closest pairs:")
for s, a, b in pairs[:5]:
    print(f"  {s:+.3f}  {a} ↔ {b}")

opposite = ["INDIRECT-UPREGULATOR", "INDIRECT-DOWNREGULATOR"]
if all(l in labels for l in opposite):
    i, j = labels.index(opposite[0]), labels.index(opposite[1])
    print(f"\nGATE: {opposite[0]} ↔ {opposite[1]} cosine = {sim[i,j].item():+.3f}")
    print(f"  v4 baseline: +0.936; v5 acceptance: < +0.7")
    print(f"  -> {'PASS' if sim[i,j].item() < 0.7 else 'FAIL'}")
