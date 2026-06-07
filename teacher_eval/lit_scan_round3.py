"""Round-3 broad lit-scan: cover angles not yet hit in round 1/2.

Specifically targeting:
- generative classifier calibration / verbalizer / first-token logit
- constrained decoding / structured generation for IE
- conformal prediction / selective generation for LLM
- KG / ontology / UMLS augmented IE
- LoRA / PEFT generalization / forgetting
- mixture-of-experts / multi-corpus separation
- curriculum / pre-finetune / two-stage IT
- representation alignment / DANN-style DA
- hard negative / contrastive RE
- long-tail / balanced softmax / decoupled training
- energy-based OOD / OOD detection in NLP
- causal feature / invariance / IRM
- self-consistency / sampling confidence
- test-time training / online adapt for LLM
- coreset / data selection / mixture ratio
- doc-level RE / longer context coreference
- 2025-2026 BioRE latest
- counterfactual augmentation / RE-DA
- prompt distribution matching
- biomed entity linking + RE joint
"""
import json
from pathlib import Path
from deepxiv_sdk import Reader

TOKEN = open("/root/.env").read().split("=", 1)[1].strip()
reader = Reader(token=TOKEN)

DATE_FROM = "2024-06-01"
DATE_TO = "2026-06-30"

QUERIES = [
    # 1. Generative classifier / verbalizer / first-token calibration
    ("verbalizer_calibration", "verbalizer calibration first-token probability LLM classification"),
    ("generative_classifier_bias", "generative classifier surface form bias LLM probability"),
    ("contextual_calibration_LLM", "contextual calibration LLM zero-shot few-shot"),

    # 2. Constrained / structured generation for IE
    ("constrained_decoding_IE", "constrained decoding structured generation information extraction"),
    ("structured_output_LLM_IE", "structured output LLM information extraction JSON schema"),
    ("guided_decoding_NER_RE", "guided decoding NER relation extraction grammar"),

    # 3. Conformal prediction / selective generation
    ("conformal_LLM", "conformal prediction LLM generation coverage"),
    ("selective_generation", "selective generation LLM abstention IE"),

    # 4. KG / ontology augmented
    ("UMLS_RE", "UMLS knowledge graph augmented biomedical relation extraction"),
    ("ontology_grounded_IE", "ontology grounded information extraction biomedical"),
    ("entity_linking_RE_joint", "entity linking joint relation extraction biomedical"),

    # 5. LoRA / PEFT
    ("lora_generalization", "LoRA generalization forgetting domain shift"),
    ("peft_multi_task", "parameter efficient fine-tuning multi-task transfer NLP"),
    ("lora_merging", "LoRA merging multi-domain instruction"),

    # 6. MoE / multi-corpus
    ("moe_multi_domain", "mixture of experts multi-domain instruction tuning NLP"),
    ("dataset_expert_routing", "expert routing dataset domain instruction"),

    # 7. Curriculum / two-stage
    ("curriculum_IT_NLP", "curriculum instruction tuning NLP order"),
    ("two_stage_finetune_biomed", "two stage pretraining fine-tuning biomedical NLP"),
    ("continual_IT", "continual instruction tuning forgetting"),

    # 8. Representation alignment / DA
    ("DANN_NLP", "domain adversarial training NLP cross-domain extraction"),
    ("invariant_representation_NLP", "invariant representation learning NLP domain generalization"),

    # 9. Hard negative / contrastive
    ("hard_negative_RE", "hard negative mining relation extraction contrastive"),
    ("contrastive_RE_LLM", "contrastive learning relation extraction LLM"),

    # 10. Long-tail classics applied to LLM
    ("balanced_softmax_recent", "balanced softmax long-tail classification recent 2025"),
    ("decoupled_classifier_LLM", "decoupled representation classifier long-tail LLM"),
    ("LDAM_logit_adjust", "label distribution aware margin LDAM logit adjustment"),

    # 11. OOD detection in NLP
    ("OOD_detection_NLP", "out-of-distribution detection NLP classification energy"),
    ("ood_text_classification", "out-of-distribution text classification 2025"),

    # 12. Causal / invariance
    ("IRM_NLP", "invariant risk minimization NLP cross-domain"),
    ("causal_invariance_RE", "causal invariance feature relation extraction"),

    # 13. Self-consistency / sampling confidence
    ("self_consistency_LLM_IE", "self consistency LLM information extraction uncertainty"),
    ("sampling_confidence_LLM", "sampling confidence LLM structured prediction"),

    # 14. Test-time training / online
    ("TTT_LLM", "test time training LLM language model adaptation"),
    ("online_adaptation_LLM_IE", "online adaptation LLM information extraction"),

    # 15. Data selection / mixture
    ("data_selection_IT", "data selection instruction tuning mixture ratio"),
    ("coreset_finetune", "coreset selection fine-tuning LLM"),

    # 16. Doc-level RE
    ("doc_level_RE_2025", "document level relation extraction 2025 LLM"),
    ("longctx_BioRE", "long context biomedical relation extraction coreference"),

    # 17. Latest 2025-2026 BioRE specifically
    ("biored_2026", "BioRED biomedical relation extraction 2026"),
    ("biomed_RE_latest", "biomedical relation extraction document multi-corpus 2025"),

    # 18. Counterfactual / RE-DA
    ("counterfactual_RE", "counterfactual data augmentation relation extraction"),
    ("re_data_aug_LLM", "relation extraction data augmentation LLM synthesize"),

    # 19. Prompt matching
    ("prompt_distribution_match", "prompt distribution matching calibration classification"),

    # 20. Distillation specifically
    ("distill_biomed_LLM", "knowledge distillation biomedical LLM small model"),

    # 21. NL gen for low-resource RE
    ("synthetic_RE_LLM", "synthetic data LLM relation extraction low resource"),

    # 22. Posterior calibration in IE
    ("posterior_calib_IE", "posterior calibration information extraction class probability"),
]

results = {}
for tag, q in QUERIES:
    try:
        r = reader.search(q, size=6, date_from=DATE_FROM, date_to=DATE_TO)
        papers = r.get("result", [])
        results[tag] = {"query": q, "n": len(papers), "papers": [{
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "year": p.get("year") or (p.get("published_at", "") or "")[:10],
            "abstract": (p.get("abstract") or p.get("summary") or "")[:350],
        } for p in papers]}
        print(f"  {tag:<28} {len(papers)} hits")
    except Exception as e:
        print(f"  {tag:<28} ERROR: {e}")
        results[tag] = {"query": q, "error": str(e)}

out = Path("teacher_eval/results/lit_scan_round3.json")
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nWrote {out}")

print("\n" + "="*100)
for tag, d in results.items():
    if "papers" not in d:
        continue
    print(f"\n### {tag}   query: {d['query']}")
    for p in d["papers"][:4]:
        date = p["year"][:10] if p["year"] else "????-??-??"
        print(f"  [{date}] {p['arxiv_id'] or '?':<14} {p['title'][:90]}")
