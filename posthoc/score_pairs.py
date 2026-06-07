"""Score every (entity_pair, candidate_label) with the LLM's sequence logprob.

For each entity pair (i, j) in a doc:
  1. Build a per-pair prompt (training-format, single target in Questions).
  2. For each candidate label in [None] + rel_list:
       Compute log P(answer_string | prompt) by feeding prompt+answer to
       vLLM with prompt_logprobs=1 and summing logprobs over the answer tokens.
  3. Save to JSON: doc -> pair -> {logprobs: [n_labels], gold: int, h_type, t_type}

The resulting JSON drives all post-hoc methods (P2P / LA / PAS / TECP).

GPU required. Reads variant D LoRA from --variant_dir.

Output: <output_dir>/<dataset_basename>_scores.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepro import read_biored
from utils import feature2text, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./dataset/biored")
    p.add_argument("--dev_file", default="processed_test.pubtator")
    p.add_argument("--test_file", default="processed_bc8_test.pubtator")
    p.add_argument("--ood_test_files", default="",
                   help="Comma-separated pubtator paths for OOD eval (cdr / disgenet / pharmgkb).")
    p.add_argument("--model_name_or_path",
                   default="base_models/Qwen3-8B-Base", type=str)
    p.add_argument("--prepro_tokenizer_path", default="bert-base-uncased", type=str)
    p.add_argument("--variant_dir", required=True,
                   help="e.g. results/biored_finetune/D/checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--extract_prompt_file", default="./meta/baseline/extract.txt")
    p.add_argument("--max_seq_length", default=3072, type=int)
    p.add_argument("--use_direction", action="store_true")
    p.add_argument("--vllm_batch_size", default=256, type=int)
    p.add_argument("--vllm_max_num_seqs", default=64, type=int)
    p.add_argument("--vllm_gpu_mem_util", default=0.80, type=float)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--ood_force_dataset_name", type=str, default="BioRED")
    p.add_argument("--limit_docs", type=int, default=-1,
                   help="-1 for all; useful for quick smoke test")
    return p.parse_args()


def build_pair_prompt(extract_template, doc_text, dataset_name, rel_list_str,
                      source_marker, target_marker, use_direction):
    """One source/target pair per prompt. Output line is always '1. <label>'."""
    phrase = (f'from {source_marker} to' if use_direction
              else f'between {source_marker} and')
    questions = f"What is the relation {phrase} the following entities?\n1.{target_marker}\n"
    prompt = extract_template.replace('[Input Text]', doc_text)
    prompt = prompt.replace('[Relation List]', rel_list_str)
    prompt = prompt.replace('[Questions]', questions)
    prompt = prompt.replace('[Dataset]', dataset_name or 'BioRED')
    return prompt


def build_pair_jobs(args, features, extract_template):
    """Walk each doc, emit per-pair prompts + candidate label list + gold.

    Each emitted job has:
      doc_idx, h, t, h_type, t_type, gold_rel_id, prompt, candidates, rel_list,
      dataset_name
    gold_rel_id: 0 = None, 1..len(rel_list) = positive label.
    """
    jobs = []
    for doc_idx, feature in enumerate(features):
        ent_types = feature.get('entity_types', []) or []
        original_doc, entity_names = feature2text(
            args, feature['input_ids'], feature['entity_pos'],
            aug_rate=0, entity_types=ent_types if ent_types else None,
        )
        if not ent_types:
            ent_types = ['Unknown'] * len(entity_names)

        rel_list = feature['rel_list']
        rel_list_str = '\n'.join(rel_list)
        dataset_name = feature.get('dataset_name', 'BioRED')

        labels = np.asarray(feature['labels'])
        gold_map = {}
        for local_idx, ht in enumerate(feature['hts']):
            h, t = int(ht[0]), int(ht[1])
            nz = np.nonzero(labels[local_idx])[0]
            if len(nz) == 0 or (len(nz) == 1 and nz[0] == 0):
                gold_map[(h, t)] = 0
            else:
                for r in nz:
                    if r != 0:
                        gold_map[(h, t)] = int(r)
                        break

        def _etype(idx):
            return ent_types[idx] if idx < len(ent_types) else 'Unknown'

        full_markers = [f'{{{i}|{entity_names[i]}|{_etype(i)}}}' for i in range(len(entity_names))]
        short_markers = [f'{{{i}}}' for i in range(len(entity_names))]

        candidates = ['None'] + list(rel_list)
        for i in range(len(entity_names)):
            for j in range(len(entity_names)):
                if i == j:
                    continue
                prompt = build_pair_prompt(
                    extract_template, original_doc, dataset_name, rel_list_str,
                    full_markers[i], short_markers[j], args.use_direction,
                )
                gold = gold_map.get((i, j), 0)
                jobs.append({
                    'doc_idx': doc_idx,
                    'h': i, 't': j,
                    'h_type': _etype(i), 't_type': _etype(j),
                    'gold_rel_id': gold,
                    'prompt': prompt,
                    'candidates': candidates,
                    'rel_list': list(rel_list),
                    'dataset_name': dataset_name,
                })
    return jobs


def build_chat_prompt(prompt_text, tokenizer):
    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt_text


def score_jobs(args, jobs):
    """For each job, run len(candidates) scoring calls with prompt_logprobs.

    We construct full = (chat_prompt + answer_str) and sum logprobs over the
    answer-token range. Boundary is found via tokenize(chat_prompt).
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_name_or_path,
        dtype='bfloat16',
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_lora=True,
        max_model_len=int(args.max_seq_length) + 64,
        enable_prefix_caching=True,
        disable_log_stats=True,
        gpu_memory_utilization=float(args.vllm_gpu_mem_util),
        max_num_seqs=int(args.vllm_max_num_seqs),
        enforce_eager=False,
    )
    lora_request = LoRARequest("lora", 1, args.variant_dir)

    answer_template = "1. {label}\n"
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )

    # Flatten: each scoring task = (job_idx, label_idx, prompt_token_ids, boundary)
    requests = []
    for job_idx, job in enumerate(jobs):
        chat_prompt = build_chat_prompt(job['prompt'], tokenizer)
        prefix_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
        for li, label in enumerate(job['candidates']):
            answer_ids = tokenizer.encode(answer_template.format(label=label),
                                          add_special_tokens=False)
            requests.append({
                'job_idx': job_idx,
                'label_idx': li,
                'prompt_token_ids': prefix_ids + answer_ids,
                'boundary': len(prefix_ids),
            })

    # Init per-job result containers
    out_scores = [{lab: float('nan') for lab in job['candidates']} for job in jobs]

    if jobs:
        print(f"Total scoring requests: {len(requests)} "
              f"({len(jobs)} pairs × {len(jobs[0]['candidates'])} labels)")
    else:
        print("no jobs")
        return out_scores

    BATCH = max(8, int(args.vllm_batch_size))
    for start in tqdm(range(0, len(requests), BATCH), desc='score'):
        batch = requests[start:start + BATCH]
        vllm_inputs = [{"prompt_token_ids": r['prompt_token_ids']} for r in batch]
        outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        for r, out in zip(batch, outputs):
            full_ids = list(out.prompt_token_ids)
            prompt_lps = out.prompt_logprobs  # list[Optional[dict]] same length
            boundary = r['boundary']
            score = 0.0
            for i in range(boundary, len(full_ids)):
                lp_entry = prompt_lps[i]
                if lp_entry is None:
                    continue
                tid = full_ids[i]
                if tid in lp_entry:
                    score += float(lp_entry[tid].logprob)
                else:
                    score += -100.0  # forced token unlikely
            job_idx = r['job_idx']
            label = jobs[job_idx]['candidates'][r['label_idx']]
            out_scores[job_idx][label] = score
    return out_scores


def main():
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args)

    if "docred" in args.data_dir.lower():
        args.dataset = 'docred'
    elif "biomed" in args.data_dir.lower() or "biored" in args.data_dir.lower():
        args.dataset = 'biored'
    else:
        args.dataset = 'biored'

    args.prepro_tokenizer = AutoTokenizer.from_pretrained(args.prepro_tokenizer_path)
    args.max_input_len = 0
    args.extract_prompt = open(args.extract_prompt_file).read()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.ood_test_files:
        for p in args.ood_test_files.split(','):
            p = p.strip()
            if p:
                targets.append(('ood', p))
    else:
        targets.append(('dev', os.path.join(args.data_dir, args.dev_file)))
        targets.append(('test', os.path.join(args.data_dir, args.test_file)))

    for kind, fp in targets:
        basename = os.path.splitext(os.path.basename(fp))[0]
        out_path = out_dir / f"{basename}_scores.json"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue
        print(f"\n=== Scoring {kind}: {fp} -> {out_path} ===")
        features = read_biored(fp, args.prepro_tokenizer,
                               max_seq_length=args.max_seq_length,
                               max_samples=None, use_direction=args.use_direction)
        if not features:
            print(f"  no docs parsed, skipping")
            continue
        if args.limit_docs > 0:
            features = features[:args.limit_docs]
            print(f"  limited to first {len(features)} docs")
        # Wrap features as expected (test_llm runs DataLoader collate; we need a
        # single-doc-per-feature representation, so reuse what features come in).
        # The features list from read_biored already has per-doc dicts.

        jobs = build_pair_jobs(args, features, args.extract_prompt)
        print(f"  pairs to score: {len(jobs)}")
        if not jobs:
            print(f"  no pairs, skipping")
            continue
        scores = score_jobs(args, jobs)

        # Group results by doc
        docs_out = {}
        for job, sc in zip(jobs, scores):
            d = docs_out.setdefault(job['doc_idx'], {
                'doc_idx': job['doc_idx'],
                'dataset_name': job['dataset_name'],
                'rel_list': job['rel_list'],
                'pairs': [],
            })
            d['pairs'].append({
                'h': job['h'], 't': job['t'],
                'h_type': job['h_type'], 't_type': job['t_type'],
                'gold_rel_id': job['gold_rel_id'],
                'logprobs': [sc[lab] for lab in job['candidates']],
                'candidates': job['candidates'],
            })

        payload = {
            'source_file': fp,
            'variant_dir': args.variant_dir,
            'use_direction': args.use_direction,
            'n_docs': len(docs_out),
            'n_pairs': len(jobs),
            'docs': [docs_out[k] for k in sorted(docs_out)],
        }
        out_path.write_text(json.dumps(payload))
        print(f"  saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
