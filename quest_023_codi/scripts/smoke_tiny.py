"""50-step smoke run: verify no NaN, memory OK, loss curves sane."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from src.codi_model import load_codi_trainable
from src.codi_data import CODIDataset, make_codi_collator
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

DATA = "/tmp/synth_smoke/training_v4/synth_train.jsonl"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = "/home/ds/quest_023_codi/artifacts/mean.pt"
N_STEPS = 50

print("Loading model...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
# Subsample 200 records for tiny smoke
ds = CODIDataset(DATA, label_list=labels, sample_indices=list(range(200)))
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=make_codi_collator())

opt_params = [p for p in model.parameters() if p.requires_grad] + list(latents.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
print(f"  trainable params: {sum(p.numel() for p in opt_params)}", flush=True)

embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
cache = CODILabelCache(label_dict, embs_all)
print(f"  label cache: {tuple(embs_all.shape)}", flush=True)

cfg = CODILossConfig(k=4, warmup_steps=10, ramp_steps=10)  # short warmup for tiny smoke
rng = random.Random(0)
model.train()

t0 = time.time()
step = 0
while step < N_STEPS:
    if step % 10 == 0 and step > 0:
        with torch.no_grad():
            embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
            cache = CODILabelCache(label_dict, embs_all)
    for batch in loader:
        if step >= N_STEPS: break
        label_embs = cache.sample_per_label(rng).to("cuda")
        loss, log = codi_step(model, backbone, latents, tok, ids, batch, label_embs,
                                 mean.to("cuda"), cfg, step)
        if not torch.isfinite(loss):
            print(f"NaN at step {step}: {log}"); sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        if step % 5 == 0 or step == N_STEPS - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:3d}: total={loss.item():.3f} L_cls_S={log['L_cls_S']:.3f} "
                  f"acc_S={log['acc_S']:.2f} acc_T={log['acc_T']:.2f} "
                  f"align_gap={log['align_gap']:.3f} mem={mem:.1f}GB", flush=True)
        step += 1
elapsed = time.time() - t0
print(f"\nDONE: {step} steps in {elapsed:.0f}s ({elapsed/step:.1f}s/step)", flush=True)
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
