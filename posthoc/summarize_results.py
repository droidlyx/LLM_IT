"""Build paper-ready summary table + EXPERIMENTS.md snippet from *_adjusted.json.

Usage:
  python posthoc/summarize_results.py --results_dir posthoc/results --out posthoc/results/SUMMARY.md
"""
import argparse
import json
from pathlib import Path


DATASET_LABEL_MAP = {
    "processed_test": "BioRED dev (processed_test)",
    "processed_bc8_test": "BC8 test",
    "cdr": "cdr (BC5CDR)",
    "disgenet": "disgenet",
    "pharmgkb": "pharmgkb",
}

DATASET_ORDER = ["processed_test", "processed_bc8_test", "cdr", "disgenet", "pharmgkb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="posthoc/results")
    p.add_argument("--out", default="posthoc/results/SUMMARY.md")
    return p.parse_args()


def best_method(d: dict) -> tuple[str, float]:
    base_f1 = d["baseline"]["micro"]["f1"]
    best_m, best_f1 = "baseline", base_f1
    for m, v in d.items():
        if m == "baseline" or "micro" not in v:
            continue
        if v["micro"]["f1"] > best_f1:
            best_m, best_f1 = m, v["micro"]["f1"]
    return best_m, best_f1


def main():
    args = parse_args()
    rd = Path(args.results_dir)
    files = {}
    for f in rd.glob("*_adjusted.json"):
        stem = f.stem.replace("_adjusted", "")
        files[stem] = f

    out_lines = ["# Phase 1 Post-hoc Calibration — Full Results", ""]
    out_lines.append("Variant: D (Qwen3-8B-Base + LoRA, BioRED-only + loss reweight)")
    out_lines.append("Methods: P2P / LA / PAS / TECP / P2P+TECP")
    out_lines.append("")
    out_lines.append("## Micro F1 summary")
    out_lines.append("")
    out_lines.append("| Dataset | n_pairs | Baseline F1 | Best F1 | Δ | Method |")
    out_lines.append("|---|---|---|---|---|---|")

    for ds in DATASET_ORDER:
        if ds not in files:
            continue
        d = json.load(open(files[ds]))
        # n_pairs: re-load the corresponding scores JSON
        scores_path = rd / f"{ds}_scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                n_pairs = json.load(f)["n_pairs"]
        else:
            n_pairs = "?"
        base_f1 = d["baseline"]["micro"]["f1"]
        method, best = best_method(d)
        delta = best - base_f1
        label = DATASET_LABEL_MAP.get(ds, ds)
        out_lines.append(f"| {label} | {n_pairs} | {base_f1:.4f} | **{best:.4f}** | **{delta:+.4f}** | {method} |")

    out_lines.append("")
    out_lines.append("## Method ranking by mean Δ across 5 datasets")
    out_lines.append("")

    method_deltas = {}
    for ds in DATASET_ORDER:
        if ds not in files:
            continue
        d = json.load(open(files[ds]))
        base_f1 = d["baseline"]["micro"]["f1"]
        for m, v in d.items():
            if m == "baseline" or "micro" not in v:
                continue
            method_deltas.setdefault(m, []).append(v["micro"]["f1"] - base_f1)

    rows = []
    for m, deltas in method_deltas.items():
        mean = sum(deltas) / max(1, len(deltas))
        rows.append((m, mean, len(deltas), deltas))
    rows.sort(key=lambda r: -r[1])

    out_lines.append("| Method | Mean Δ | n datasets | per-dataset Δ |")
    out_lines.append("|---|---|---|---|")
    for m, mean, n, deltas in rows:
        deltas_str = ", ".join(f"{d:+.3f}" for d in deltas)
        out_lines.append(f"| {m} | **{mean:+.4f}** | {n} | {deltas_str} |")

    out_lines.append("")
    out_lines.append("## Per-class breakdown (best method per dataset)")
    out_lines.append("")
    for ds in DATASET_ORDER:
        if ds not in files:
            continue
        d = json.load(open(files[ds]))
        method, _ = best_method(d)
        if method == "baseline" or method not in d or "per_class" not in d[method]:
            continue
        out_lines.append(f"### {DATASET_LABEL_MAP.get(ds, ds)} (best={method})")
        out_lines.append("")
        out_lines.append("| Class | Baseline P/R/F1 | Best P/R/F1 |")
        out_lines.append("|---|---|---|")
        base_pc = d["baseline"]["per_class"]
        best_pc = d[method]["per_class"]
        for cls, base_v in base_pc.items():
            best_v = best_pc.get(cls, base_v)
            out_lines.append(
                f"| {cls} | {base_v['p']:.3f} / {base_v['r']:.3f} / {base_v['f1']:.3f} | "
                f"{best_v['p']:.3f} / {best_v['r']:.3f} / {best_v['f1']:.3f} |"
            )
        out_lines.append("")

    out_lines.append("")
    out_lines.append("## Effective prior diagnostic")
    out_lines.append("")
    out_lines.append("p_eff (model's self-estimated prior) vs p_target (oracle from gold):")
    out_lines.append("")
    for ds in DATASET_ORDER:
        if ds not in files:
            continue
        d = json.load(open(files[ds]))
        if "P2P_oracle" not in d or "p_eff" not in d["P2P_oracle"]:
            continue
        p_eff = d["P2P_oracle"]["p_eff"]
        p_target = d["P2P_oracle"]["p_target"]
        out_lines.append(f"**{DATASET_LABEL_MAP.get(ds, ds)}**")
        out_lines.append(f"  - p_eff:    `{[round(x, 3) for x in p_eff]}`")
        out_lines.append(f"  - p_target: `{[round(x, 3) for x in p_target]}`")
        out_lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {out_path}")
    print()
    print("\n".join(out_lines[:30]))


if __name__ == "__main__":
    main()
