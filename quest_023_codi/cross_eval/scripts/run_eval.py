"""Zero-shot eval of quest 022 silent_cot_full model on a new dataset.

Loads:
  - Qwen3-4B-Base + LoRA adapter from quest 022 final/
  - target dataset JSONL (parsed from pubtator)
  - target dataset label_dict (5 templates per label)

Computes:
  - per-dataset mean from this dataset's own label templates (per design v2)
  - centered label embeddings (mean of 5 centered templates per label)
  - centered sample embeddings (Pass A format)
  - argmax cosine -> prediction
  - macro/micro F1, per-class breakdown
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, argparse, sys, time
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_BASE = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"
ADAPTER_DIR = "/home/ds/DeepScientist/quests/022/.ds/worktrees/idea-idea-7ffb9515/artifacts/runs/silent_cot_full/final"


def fmt_input(r):
    return (f"Document: {r['doc']}\n\n"
            f"Entity 1: {r['e1_text']} ({r['e1_type']})\n"
            f"Entity 2: {r['e2_text']} ({r['e2_type']})\n\n"
            f"Question: What is the biological relationship between Entity 1 and Entity 2?")


@torch.no_grad()
def encode_batch_last(backbone, tok, texts, max_length=2048):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")
    out = backbone(**enc, use_cache=False)
    h = out.last_hidden_state.float()
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    h_last = h[torch.arange(h.shape[0], device="cuda"), last_idx]
    return h_last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSONL test file")
    p.add_argument("--label_dict", required=True, help="JSON: label -> list of templates")
    p.add_argument("--out", required=True, help="Output dir for metrics + predictions")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--name", default="dataset")
    p.add_argument("--no_adapter", action="store_true",
                   help="Skip loading the quest 022 LoRA adapter (= raw Qwen3-4B-Base baseline)")
    p.add_argument("--adapter_dir", default=None,
                   help="Override ADAPTER_DIR (default: quest 022 final/)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading base model{'(+ adapter)' if not args.no_adapter else ' (NO adapter — raw baseline)'}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    if args.no_adapter:
        # Use the raw base model
        model.eval()
        backbone = model.model  # the inner Qwen3Model
    else:
        adapter_path = args.adapter_dir if args.adapter_dir else ADAPTER_DIR
        print(f"  adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        backbone = model.base_model.model.model
    backbone.eval()

    # Load label dict
    label_dict = json.load(open(args.label_dict))
    labels = list(label_dict.keys())
    n_labels = len(labels)
    print(f"Loaded {n_labels} labels: {labels}", flush=True)

    # Encode all templates and compute per-dataset mean
    print("Encoding label templates (computing mean)...", flush=True)
    all_template_texts = []
    label_template_idx = {}  # label -> (start, end) indices in all_template_texts
    for lbl in labels:
        s = len(all_template_texts)
        all_template_texts.extend(label_dict[lbl])
        label_template_idx[lbl] = (s, len(all_template_texts))
    # Encode in batches
    all_template_embs = []
    for i in range(0, len(all_template_texts), args.batch_size):
        batch = all_template_texts[i:i+args.batch_size]
        embs = encode_batch_last(backbone, tok, batch)
        all_template_embs.append(embs)
    all_template_embs = torch.cat(all_template_embs, dim=0).float()  # (n_templates, d)
    global_mean = all_template_embs.mean(0)
    print(f"  mean shape: {tuple(global_mean.shape)}, norm: {global_mean.norm().item():.2f}", flush=True)

    # Centered label embeddings: mean of 5 centered templates per label
    label_embs = []
    for lbl in labels:
        s, e = label_template_idx[lbl]
        block = all_template_embs[s:e] - global_mean  # centered (5, d)
        emb = block.mean(0)
        label_embs.append(emb)
    label_embs = torch.stack(label_embs).float()  # (n_labels, d)
    label_embs_norm = label_embs / (label_embs.norm(dim=-1, keepdim=True) + 1e-9)

    # G1 sanity: check intra/inter gap
    print("\nG1 sanity check (centering gap):", flush=True)
    intra_vals = []
    for lbl in labels:
        s, e = label_template_idx[lbl]
        block = all_template_embs[s:e] - global_mean
        block_n = block / (block.norm(dim=-1, keepdim=True) + 1e-9)
        C = (block_n @ block_n.T)
        n = C.shape[0]
        intra = (C.sum() - C.trace()).item() / (n*(n-1))
        intra_vals.append(intra)
    inter_M = label_embs_norm @ label_embs_norm.T
    n = inter_M.shape[0]
    inter = (inter_M.sum() - inter_M.trace()).item() / (n*(n-1))
    print(f"  intra={np.mean(intra_vals):+.3f}  inter={inter:+.3f}  gap={np.mean(intra_vals)-inter:+.3f}", flush=True)

    # Load test records
    records = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") in labels:
                records.append(r)
    print(f"\nLoaded {len(records)} eval records ({Counter(r['label'] for r in records).most_common()})", flush=True)

    # Eval loop
    print(f"Running eval (batch_size={args.batch_size})...", flush=True)
    preds = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i+args.batch_size]
        texts = [fmt_input(r) for r in batch]
        h = encode_batch_last(backbone, tok, texts).float()
        h_c = h - global_mean
        h_n = h_c / (h_c.norm(dim=-1, keepdim=True) + 1e-9)
        scores = h_n @ label_embs_norm.T   # (B, n_labels)
        pred_idx = scores.argmax(dim=-1).cpu().numpy()
        for r, pi, s in zip(batch, pred_idx, scores.cpu().numpy()):
            preds.append({**r, "pred": labels[pi], "scores": s.tolist()})
        if (i // args.batch_size) % 20 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            eta = elapsed / max(done, 1) * (len(records) - done)
            print(f"  {done}/{len(records)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # Metrics
    gold_arr = [r["label"] for r in preds]
    pred_arr = [r["pred"] for r in preds]

    # Per-class
    per_class = {}
    f1s = []
    for lbl in labels:
        g = sum(1 for x in gold_arr if x == lbl)
        p = sum(1 for x in pred_arr if x == lbl)
        tp = sum(1 for r in preds if r["label"] == lbl and r["pred"] == lbl)
        prec = tp/p if p else 0
        rec = tp/g if g else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        per_class[lbl] = {"support": g, "predicted": p, "tp": tp,
                          "precision": prec, "recall": rec, "f1": f1}
        if g > 0:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    acc = sum(1 for r in preds if r["label"] == r["pred"]) / len(preds)
    micro_f1 = acc  # for single-label classification, micro-F1 = accuracy

    metrics = {
        "name": args.name,
        "n_examples": len(preds),
        "n_labels": n_labels,
        "raw": {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "accuracy": acc,
            "per_class": per_class,
        },
        "g1_centering_gap": {
            "intra": float(np.mean(intra_vals)),
            "inter": inter,
            "gap": float(np.mean(intra_vals)) - inter,
        },
    }

    # Save
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.out}/predictions.jsonl", "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Final metrics ({args.name}) ===")
    print(f"  macro-F1: {macro_f1:.4f}")
    print(f"  micro-F1: {micro_f1:.4f}")
    print(f"  per-class:")
    for lbl, m in per_class.items():
        print(f"    {lbl:<20} support={m['support']:>4} F1={m['f1']:.3f}")
    print(f"\nSaved metrics + predictions to {args.out}/")


if __name__ == "__main__":
    main()
