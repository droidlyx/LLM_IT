"""score_pairs_v3: multi-pair logprob scoring — matches deployment distribution.

Key difference vs v2 (single-pair) and v1 (per-cand sequence scoring):

v3 uses the SAME multi-pair prompt as test_llm.py:
    "1.{ent2}\n2.{ent3}\n3.{ent4}\n..." (one source × N targets)
generates "1. Association\n2. None\n3. Positive_Correlation\n..." in one
shot with logprobs=K, and at each "{N}. " boundary captures the model's
top-K logprobs at the FIRST-token-of-label position.

These logprobs are conditioned on the model's own previously-generated
answers (autoregressive — same as deployment). Calibration applied to
these logprobs yields paper-grade Δ that mirrors deployment behaviour.

Caveats:
- Exposure bias (1st-order): if post-hoc adjustment flips position N,
  positions N+1..end were conditioned on the original (un-flipped)
  prefix. We don't re-decode. In practice the impact is small because
  flips happen on borderline cases.
- If model drifts (extra labels, missing labels), some pairs won't have
  scores; we mark them with NaN and they fall back to argmax=None in
  eval_adjusted.

JSON output schema is identical to v1/v2 so eval_adjusted works unchanged.
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
    p.add_argument("--ood_test_files", default="")
    p.add_argument("--model_name_or_path", default="/root/gpufree-data/Qwen3-8B-Base", type=str)
    p.add_argument("--prepro_tokenizer_path", default="bert-base-uncased", type=str)
    p.add_argument("--variant_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--extract_prompt_file", default="./meta/baseline/extract.txt")
    p.add_argument("--max_seq_length", default=1792, type=int)
    p.add_argument("--max_new_tokens", default=512, type=int)
    p.add_argument("--use_direction", action="store_true")
    p.add_argument("--vllm_batch_size", default=256, type=int)
    p.add_argument("--vllm_max_num_seqs", default=64, type=int)
    p.add_argument("--vllm_gpu_mem_util", default=0.80, type=float)
    p.add_argument("--logprobs_k", default=20, type=int)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=-1)
    return p.parse_args()


def build_multipair_prompt(args, feature):
    """Same prompt builder as utils.construct_llm_input but per-doc-and-source-entity.

    For each source entity i, returns list of (j, prompt) where each prompt
    asks for relations between i and ALL j != i in the standard multi-pair
    format the model was trained on.

    Returns: list of dicts {h, candidates, pair_jts, prompt, gold_map}
    where pair_jts is the ordered list of target j's queried.
    """
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
    use_direction = args.use_direction

    queries = []
    for i in range(len(entity_names)):
        # Build list of targets (all j != i)
        pair_jts = [j for j in range(len(entity_names)) if j != i]
        if not pair_jts:
            continue
        # Build "Questions" section: numbered list of targets
        ent1 = full_markers[i]
        phrase = f'from {ent1} to' if use_direction else f'between {ent1} and'
        questions_parts = [f'What is the relation {phrase} the following entities?\n']
        for line_idx, j in enumerate(pair_jts, 1):
            questions_parts.append(f'{line_idx}.{short_markers[j]}\n')
        questions = ''.join(questions_parts)

        prompt = args.extract_prompt
        prompt = prompt.replace('[Input Text]', original_doc)
        prompt = prompt.replace('[Relation List]', rel_list_str)
        prompt = prompt.replace('[Questions]', questions)
        prompt = prompt.replace('[Dataset]', dataset_name or 'BioRED')

        queries.append({
            'h': i,
            'pair_jts': pair_jts,
            'gold_per_t': [gold_map.get((i, j), 0) for j in pair_jts],
            'h_type': _etype(i),
            't_types': [_etype(j) for j in pair_jts],
            'candidates': candidates,
            'rel_list': list(rel_list),
            'dataset_name': dataset_name,
            'prompt': prompt,
        })
    return queries


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


def compute_candidate_first_tokens(candidates, tokenizer):
    """Same as v2 — find first-token id when label follows '1. ' prefix.

    For multi-pair, line 2+ are preceded by '\\n2. ', '\\n3. ', etc.
    The "first token of label" should be the same regardless of line
    prefix (BPE tokenization of " Association" only depends on left
    context). Verified by check_first_tokens.py.
    """
    answer_prefix = "1. "
    prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    out = {}
    for label in candidates:
        full_ids = tokenizer.encode(answer_prefix + label, add_special_tokens=False)
        i = 0
        while i < len(prefix_ids) and i < len(full_ids) and prefix_ids[i] == full_ids[i]:
            i += 1
        out[label] = full_ids[i] if i < len(full_ids) else -1
    return out


def find_label_positions(generated_token_ids, tokenizer):
    """Walk generated tokens via state machine. Each line follows pattern:
      [\\n]? + digits + '.' + (optional whitespace tokens) + label_first_tok

    Line 1 has no leading newline (model starts with "1."). Lines 2+ start
    with "\\n". Both single-digit and double-digit line numbers are handled.

    Returns dict {line_num: token_idx_of_label_first_token}.

    State machine:
      0 = idle, looking for \\n OR (at very start) for digit
      1 = after \\n or at start, expecting digit
      2 = accumulating digits or expecting .
      3 = after ., skipping whitespace tokens, will capture first non-ws
    """
    positions = {}
    state = 1  # at start, expect digit
    line_num_buf = ""
    for tok_idx, tid in enumerate(generated_token_ids):
        s = tokenizer.decode([tid])
        if state == 0:
            if s == "\n":
                state = 1; line_num_buf = ""
        elif state == 1:
            if s.isdigit():
                line_num_buf = s; state = 2
            elif s == "\n":
                line_num_buf = ""
            else:
                state = 0
        elif state == 2:
            if s.isdigit():
                line_num_buf += s
            elif s == ".":
                state = 3
            elif s == "\n":
                state = 1; line_num_buf = ""
            else:
                state = 0
        elif state == 3:
            if s.strip() == "":
                continue  # whitespace token, skip
            try:
                ln = int(line_num_buf)
                positions[ln] = tok_idx
            except ValueError:
                pass
            state = 0
            if s == "\n":
                state = 1; line_num_buf = ""
    return positions


def score_jobs_v3(args, queries_per_doc):
    """queries_per_doc: list of (doc_idx, query) tuples. Each query has 'pair_jts'."""
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
        max_model_len=int(args.max_seq_length) + int(args.max_new_tokens) + 16,
        enable_prefix_caching=True,
        disable_log_stats=True,
        gpu_memory_utilization=float(args.vllm_gpu_mem_util),
        max_num_seqs=int(args.vllm_max_num_seqs),
        enforce_eager=False,
    )
    lora_request = LoRARequest("lora", 1, args.variant_dir)

    sampling_params = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=0.0,
        logprobs=int(args.logprobs_k),
    )

    # Pre-compute first-token IDs per unique candidate set
    cand_tok_cache = {}

    # Build requests — no '1. ' pre-pend; model generates "1. {label}\n..." naturally
    requests = []
    for entry in queries_per_doc:
        doc_idx, query = entry['doc_idx'], entry['query']
        cand_key = tuple(query['candidates'])
        if cand_key not in cand_tok_cache:
            cand_tok_cache[cand_key] = compute_candidate_first_tokens(
                query['candidates'], tokenizer
            )
        chat_prompt = build_chat_prompt(query['prompt'], tokenizer)
        chat_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
        requests.append({
            'doc_idx': doc_idx,
            'query': query,
            'prompt_token_ids': chat_ids,
        })

    # Pre-allocate per-doc outputs: doc_idx -> { (h, t): {label: logprob} }
    all_pair_scores = {}

    print(f"Total multi-pair queries: {len(requests)}  "
          f"(one per source entity)")

    BATCH = max(1, int(args.vllm_batch_size) // 4)  # each generation longer
    n_drift_total = 0
    n_pairs_covered = 0
    n_pairs_total = 0
    for start in tqdm(range(0, len(requests), BATCH), desc='score_v3'):
        batch = requests[start:start + BATCH]
        vllm_inputs = [{"prompt_token_ids": r['prompt_token_ids']} for r in batch]
        outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        for r, out in zip(batch, outputs):
            doc_idx = r['doc_idx']
            query = r['query']
            cand_first = cand_tok_cache[tuple(query['candidates'])]
            pair_jts = query['pair_jts']

            generated_ids = list(out.outputs[0].token_ids)
            generated_logprobs = out.outputs[0].logprobs  # list[dict]
            line_positions = find_label_positions(generated_ids, tokenizer)

            pair_dict = all_pair_scores.setdefault(doc_idx, {})
            for line_num, j in enumerate(pair_jts, 1):
                tok_pos = line_positions.get(line_num)
                if tok_pos is None or tok_pos >= len(generated_logprobs):
                    # Drift: model didn't produce this many lines
                    n_drift_total += 1
                    continue
                top_k = generated_logprobs[tok_pos]
                pair_scores = {}
                for label in query['candidates']:
                    tid = cand_first.get(label, -1)
                    lp_entry = top_k.get(tid) if tid >= 0 else None
                    pair_scores[label] = float(lp_entry.logprob) if lp_entry is not None else -100.0
                pair_dict[(query['h'], j)] = {
                    'logprobs': [pair_scores[lab] for lab in query['candidates']],
                    'candidates': list(query['candidates']),
                    'gold_rel_id': query['gold_per_t'][line_num - 1],
                    'h_type': query['h_type'],
                    't_type': query['t_types'][line_num - 1],
                }
                n_pairs_covered += 1
            n_pairs_total += len(pair_jts)

    print(f"\nCoverage: {n_pairs_covered}/{n_pairs_total} pairs scored "
          f"({n_pairs_covered/max(1,n_pairs_total)*100:.2f}%); drift fallback: {n_drift_total}")
    return all_pair_scores


def main():
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args)

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
            print(f"  [skip] {out_path} exists")
            continue
        print(f"\n=== {kind}: {fp} → {out_path} ===")
        features = read_biored(fp, args.prepro_tokenizer,
                               max_seq_length=args.max_seq_length,
                               max_samples=None, use_direction=args.use_direction)
        if not features:
            continue
        if args.limit_docs > 0:
            features = features[:args.limit_docs]
            print(f"  limited to first {len(features)} docs")

        all_queries = []
        for doc_idx, feature in enumerate(features):
            for query in build_multipair_prompt(args, feature):
                all_queries.append({'doc_idx': doc_idx, 'query': query})
        print(f"  multi-pair queries to generate: {len(all_queries)}")
        if not all_queries:
            continue
        pair_scores = score_jobs_v3(args, all_queries)

        # Materialize JSON in the v1/v2 schema (docs[].pairs[])
        rel_list = features[0]['rel_list']
        candidates = ['None'] + list(rel_list)
        dataset_name = features[0].get('dataset_name', 'BioRED')

        docs_out = []
        n_pairs = 0
        for doc_idx in sorted(pair_scores.keys()):
            pair_dict = pair_scores[doc_idx]
            pairs = []
            for (h, t), info in sorted(pair_dict.items()):
                pairs.append({
                    'h': h, 't': t,
                    'h_type': info['h_type'], 't_type': info['t_type'],
                    'gold_rel_id': info['gold_rel_id'],
                    'logprobs': info['logprobs'],
                    'candidates': info['candidates'],
                })
            docs_out.append({
                'doc_idx': doc_idx,
                'dataset_name': dataset_name,
                'rel_list': list(rel_list),
                'pairs': pairs,
            })
            n_pairs += len(pairs)

        payload = {
            'source_file': fp,
            'variant_dir': args.variant_dir,
            'use_direction': args.use_direction,
            'scoring_method': 'multi_pair_topk',
            'logprobs_k': int(args.logprobs_k),
            'n_docs': len(docs_out),
            'n_pairs': n_pairs,
            'docs': docs_out,
        }
        out_path.write_text(json.dumps(payload))
        print(f"  saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
