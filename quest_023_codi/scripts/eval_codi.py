# scripts/eval_codi.py
"""Full-RE eval for trained CODI adapter. NO positive-only — every record (incl. no_relation) is scored.

Reports macro-F1 + RE-style micro-F1 (TP/(TP+FP+FN) over positive predictions, excluding no_relation TNs).
Also supports post-hoc τ_cal threshold for no_relation prediction.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.codi_inputs import build_student_inputs, pool_positions
from src.codi_label_cache import encode_label_templates_codi
from src.codi_latents import LearnableLatents
from src.codi_data import fmt_task

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"


def load_codi_eval(adapter_dir: str, tok_dir: str, latents_pt: str, special_ids_json: str,
                    k: int = 4, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2",
                                                  trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    backbone = model.base_model.model.model
    d_model = base.config.hidden_size
    latents = LearnableLatents(k=k, d_model=d_model).to(device=device, dtype=torch.bfloat16)
    latents.load_state_dict(torch.load(latents_pt, map_location=device, weights_only=True))
    for p in latents.parameters(): p.requires_grad_(False)
    ids = json.load(open(special_ids_json))
    return model, backbone, tok, latents, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label_dict", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--tok_dir", required=True)
    p.add_argument("--latents_pt", required=True)
    p.add_argument("--special_ids", required=True)
    p.add_argument("--mean_pt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="dataset")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--tau_cal", type=float, default=0.0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading model + adapter...", flush=True)
    model, backbone, tok, latents, ids = load_codi_eval(
        args.adapter_dir, args.tok_dir, args.latents_pt, args.special_ids, k=args.k
    )
    mean = torch.load(args.mean_pt, map_location="cuda", weights_only=True).float()

    label_dict = json.load(open(args.label_dict))
    labels = list(label_dict.keys())
    print(f"  {len(labels)} labels: {labels}", flush=True)
    has_no_rel = "no_relation" in labels
    no_rel_idx = labels.index("no_relation") if has_no_rel else -1

    embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
    label_embs = embs_all.mean(dim=1)
    label_embs = torch.nn.functional.normalize(label_embs, p=2, dim=-1)

    records = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r["label"] not in labels:
                continue
            records.append(r)
    print(f"  {len(records)} records, label distribution: {Counter(r['label'] for r in records).most_common()}", flush=True)

    preds = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        texts = [fmt_task(r) for r in batch]
        with torch.inference_mode():
            s_embeds, s_mask, s_pool = build_student_inputs(texts, tok, model, latents, ids, max_task_len=600)
            out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
            h = pool_positions(out.last_hidden_state.float(), s_pool)
            h_c = h - mean.to("cuda").unsqueeze(0)
            h_n = torch.nn.functional.normalize(h_c, p=2, dim=-1)
            scores = h_n @ label_embs.t()

            if has_no_rel and args.tau_cal > 0:
                pos_mask = torch.ones(len(labels), device="cuda", dtype=torch.bool)
                pos_mask[no_rel_idx] = False
                pos_max = scores[:, pos_mask].max(dim=-1).values
                no_rel_score = scores[:, no_rel_idx]
                margin = pos_max - no_rel_score
                pred_idx = scores.argmax(dim=-1)
                pred_idx[margin < args.tau_cal] = no_rel_idx
            else:
                pred_idx = scores.argmax(dim=-1)

            scores_np = scores.cpu().numpy()
            pred_idx = pred_idx.cpu().numpy()
        for r, pi, s in zip(batch, pred_idx, scores_np):
            preds.append({**r, "pred": labels[pi], "scores": s.tolist()})
        if (i // args.batch_size) % 20 == 0:
            done = i + len(batch)
            print(f"  {done}/{len(records)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    gold = [r["label"] for r in preds]
    pred = [r["pred"] for r in preds]

    per_class = {}
    f1s_macro = []
    for lbl in labels:
        g = sum(1 for x in gold if x == lbl)
        pp = sum(1 for x in pred if x == lbl)
        tp = sum(1 for r in preds if r["label"] == lbl and r["pred"] == lbl)
        prec = tp / pp if pp else 0.0
        rec = tp / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"support": g, "predicted": pp, "tp": tp,
                          "precision": prec, "recall": rec, "f1": f1}
        if g > 0:
            f1s_macro.append(f1)
    macro_f1 = float(np.mean(f1s_macro))

    if has_no_rel:
        pos_labels = [l for l in labels if l != "no_relation"]
    else:
        pos_labels = labels
    tp_sum = sum(per_class[l]["tp"] for l in pos_labels)
    fp_sum = sum(per_class[l]["predicted"] - per_class[l]["tp"] for l in pos_labels)
    fn_sum = sum(per_class[l]["support"] - per_class[l]["tp"] for l in pos_labels)
    if tp_sum + fp_sum + fn_sum > 0:
        micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        re_micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    else:
        re_micro_f1 = 0.0

    accuracy_with_no_rel = sum(1 for r in preds if r["label"] == r["pred"]) / len(preds)

    metrics = {
        "name": args.name, "n_examples": len(preds), "n_labels": len(labels),
        "tau_cal": args.tau_cal,
        "full_re": {
            "macro_f1": macro_f1, "re_micro_f1": re_micro_f1,
            "accuracy_incl_no_rel": accuracy_with_no_rel,
            "per_class": per_class,
        },
    }
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.out}/predictions.jsonl", "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Full-RE metrics ({args.name}, tau_cal={args.tau_cal}) ===")
    print(f"  macro-F1:    {macro_f1:.4f}")
    print(f"  RE micro-F1: {re_micro_f1:.4f}  (positive labels only)")
    print(f"  accuracy:    {accuracy_with_no_rel:.4f}  (includes no_relation TNs, NOT a sub for F1)")
    for lbl, m in per_class.items():
        print(f"    {lbl:<25} sup={m['support']:>4} pred={m['predicted']:>4} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
