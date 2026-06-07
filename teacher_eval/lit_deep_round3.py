"""Deep-read the round-3 candidates that look most relevant to our problem.

Mapping to failure modes:
- P. BioRE 2025 latest             -> 2501.05155 (Concept Unification!)
- Q. KG/ontology-constrained RE    -> 2509.19057 (RELATE), 2408.06618, 2407.10021, 2604.16422
- R. Conformal / set-valued / abst -> 2509.00461 (TECP), 2603.22966, 2502.06884, 2604.18419, 2507.06867
- S. LoRA forgetting / merging     -> 2603.02224, 2509.25414, 2604.16826, 2504.17780
- T. MoE multi-domain              -> 2509.16882, 2406.11256, 2410.13458
- U. Adversarial biomed IE         -> 2509.11191 (RanAT4BIE)
- V. Counterfactual / consistency  -> 2407.06699, 2410.20710
- W. Structured output / schema    -> 2510.08623 (PARSE), 2505.04016 (SLOT)
- X. Data selection IT             -> 2601.13697, 2406.11256
- Y. CoT/abstention dynamic        -> 2604.18419
- Z. Uncertainty IE                -> 2503.00172 (survey), 2508.10036
"""
import json
from deepxiv_sdk import Reader

TOKEN = open("/root/.env").read().split("=", 1)[1].strip()
reader = Reader(token=TOKEN)

TARGETS = [
    # P. BioRE 2025 latest
    ("2501.05155", "P. Biomedical RE via Adaptive Doc-Relation Cross-Mapping + Concept Unification"),
    # Q. Ontology / KG-constrained biomed RE
    ("2509.19057", "Q. RELATE: BioRE with LLMs and Ontology Constraints"),
    ("2408.06618", "Q. Generalized KG-enhanced biomed entity+RE"),
    ("2407.10021", "Q. Doc-level Clinical Entity+RE via KB-Guided Generation"),
    ("2604.16422", "Q. Injecting Structured Biomed Knowledge: Continual Pretraining vs single-shot"),
    # R. Conformal / set-valued / abstention
    ("2509.00461", "R. TECP: Token-Entropy Conformal Prediction for LLMs"),
    ("2603.22966", "R. Set-Valued Prediction for LLMs with Coverage Guarantees"),
    ("2502.06884", "R. Learning Conformal Abstention Policies"),
    ("2604.18419", "R. Knowing When to Quit: Dynamic Abstention in LLM Reasoning"),
    ("2507.06867", "R. Conformal Prediction for Long-Tailed Classification"),
    # S. LoRA forgetting / merging
    ("2603.02224", "S. Subspace Geometry Governs Catastrophic Forgetting in LoRA"),
    ("2509.25414", "S. Rethinking Parameter Sharing for LLM FT with Multiple LoRAs"),
    ("2604.16826", "S. Crowded in B-Space: Calibrating Shared Directions for LoRA Merging"),
    # T. MoE / Expert routing
    ("2509.16882", "T. Dynamic Expert Specialization for Multi-Domain MoE"),
    ("2410.13458", "T. MedINST: Meta Dataset of Biomedical Instructions"),
    # U. Adversarial biomed IE
    ("2509.11191", "U. RanAT4BIE: Random Adversarial Training for Biomedical IE"),
    # V. Counterfactual / consistency
    ("2407.06699", "V. Consistent Doc-Level RE via Counterfactuals"),
    ("2410.20710", "V. Relation-based Counterfactual DA + Contrastive for Robust Continual RE"),
    # W. Structured output / schema
    ("2510.08623", "W. PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction"),
    # X. Data selection / mixture for IT
    ("2601.13697", "X. Uncertainty-Aware Gradient SNR Data Selection for IT"),
    ("2406.11256", "X. Dynamic Data Mixing Maximizes IT for MoE"),
    # Y. Uncertainty IE
    ("2503.00172", "Y. Survey of Uncertainty Estimation Methods on LLMs"),
    ("2508.10036", "Y. Reflect-then-Learn: Active Prompting for IE Guided by Introspective Confidence"),
    # Z. Hallucination in synth RE
    ("2410.08393", "Z. Effects of Hallucinations in Synthetic Training Data for RE"),
    ("2410.01154", "Z. Unleashing LLMs in Zero-shot RE via Self-Prompting"),
    ("2406.11162", "Z. How Good are LLMs at RE under Low-Resource"),
    # extra: LLM-IE doc-level
    ("2408.13889", "Z. LLM with Relation Classifier for Doc-Level RE"),
    # extra: BioRED 2025 directed cross-mapping (similar to ours)
    ("2412.08900", "Z. AI-assisted Knowledge Discovery in Biomed Lit"),
    # extra: continual RE LLM
    ("2508.12031", "Z. Learning Wisdom from Errors: LLM Continual RE"),
]

results = {}
for aid, hint in TARGETS:
    try:
        b = reader.brief(aid)
        results[aid] = {
            "hint": hint,
            "title": b.get("title", "?"),
            "publish_at": b.get("publish_at", "")[:10],
            "citations": b.get("citations", 0),
            "tldr": b.get("tldr", "")[:800],
            "keywords": b.get("keywords", [])[:6],
            "github": b.get("github_url", ""),
        }
        print(f"OK   {aid}  {b.get('title','?')[:70]}")
    except Exception as e:
        results[aid] = {"hint": hint, "error": str(e)}
        print(f"ERR  {aid}: {e}")

import pathlib
out = pathlib.Path("teacher_eval/results/lit_deep_round3.json")
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nWrote {out}\n")

print("="*100)
for aid, d in results.items():
    if "title" not in d:
        continue
    print(f"\n--- [{d['publish_at']}] {aid} (cites: {d.get('citations',0)}) ---")
    print(f"HINT : {d['hint']}")
    print(f"TITLE: {d['title']}")
    print(f"TLDR : {d['tldr']}")
    if d.get("github"):
        print(f"GH   : {d['github']}")
