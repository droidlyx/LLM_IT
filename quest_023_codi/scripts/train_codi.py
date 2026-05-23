# scripts/train_codi.py
"""Smoke v5: 2000 steps on 11K multi-schema synth, save adapter + latents + tokenizer."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from src.codi_model import load_codi_trainable
from src.codi_data import CODIDataset, make_codi_collator
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

OUT_DIR = "/home/ds/quest_023_codi/artifacts/smoke_v5"
DATA = "/tmp/synth_smoke/training_v4/synth_train.jsonl"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = "/home/ds/quest_023_codi/artifacts/mean.pt"
N_STEPS = 2000
CACHE_REFRESH_EVERY = 50

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()
print(f"  mean norm: {mean.norm().item():.2f}", flush=True)

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
ds = CODIDataset(DATA, label_list=labels)
print(f"  dataset: {len(ds)} samples, {len(labels)} labels", flush=True)
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=make_codi_collator())

opt_params = [p for p in model.parameters() if p.requires_grad] + list(latents.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
n_trainable = sum(p.numel() for p in opt_params)
print(f"  trainable: {n_trainable/1e6:.1f}M params", flush=True)

cfg = CODILossConfig(k=4, tau=0.07, alpha=0.5, beta=1.0, gamma=0.5,
                      warmup_steps=200, ramp_steps=200)
rng = random.Random(123)
model.train()

losses, accs_S, accs_T = [], [], []
t0 = time.time()
step = 0
embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
cache = CODILabelCache(label_dict, embs_all)

while step < N_STEPS:
    for batch in loader:
        if step >= N_STEPS: break
        if step > 0 and step % CACHE_REFRESH_EVERY == 0:
            with torch.no_grad():
                embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
                cache = CODILabelCache(label_dict, embs_all)
        label_embs = cache.sample_per_label(rng).to("cuda")
        loss, log = codi_step(model, backbone, latents, tok, ids, batch, label_embs,
                                 mean.to("cuda"), cfg, step)
        if not torch.isfinite(loss):
            print(f"NaN at step {step}: {log}"); sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        losses.append(loss.item()); accs_S.append(log["acc_S"]); accs_T.append(log["acc_T"])
        if step % 50 == 0 or step == N_STEPS - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:4d}/{N_STEPS}: total={loss.item():.3f} "
                  f"L_cls_S={log['L_cls_S']:.3f} L_cls_T={log['L_cls_T']:.3f} "
                  f"L_align={log['L_align']:.3f} L_distill={log['L_distill']:.3f} "
                  f"acc_S={log['acc_S']:.2f} acc_T={log['acc_T']:.2f} "
                  f"gap={log['align_gap']:.3f} b={log['beta_eff']:.2f} g={log['gamma_eff']:.2f} "
                  f"mem={mem:.1f}GB", flush=True)
        step += 1

print(f"\nDONE: {step} steps in {time.time()-t0:.0f}s", flush=True)
for i in range(0, len(losses), 200):
    cl = losses[i:i+200]; cas = accs_S[i:i+200]; cat = accs_T[i:i+200]
    print(f"  steps {i:4d}-{i+len(cl)-1:4d}: avg_loss={sum(cl)/len(cl):.3f} "
          f"avg_acc_S={sum(cas)/len(cas):.3f} avg_acc_T={sum(cat)/len(cat):.3f}", flush=True)

model.save_pretrained(f"{OUT_DIR}/lora_adapter")
torch.save(latents.state_dict(), f"{OUT_DIR}/latents.pt")
tok.save_pretrained(f"{OUT_DIR}/tokenizer")
with open(f"{OUT_DIR}/special_ids.json", "w") as f:
    json.dump(ids, f)
import shutil
shutil.copy(MEAN_PT, f"{OUT_DIR}/mean.pt")
print(f"\nSaved adapter+latents+tokenizer to {OUT_DIR}", flush=True)
