"""Compare vLLM greedy output WITH and WITHOUT logprobs=20."""
import os, sys, argparse, torch
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

sys.path.insert(0, '.'); sys.path.insert(0, 'posthoc')
from prepro import read_biored
from transformers import AutoTokenizer
from score_pairs_v3 import build_multipair_prompt, build_chat_prompt

def main():
    ns = argparse.Namespace(
        max_input_len=0,
        extract_prompt=open("./meta/baseline/extract.txt").read(),
        use_direction=False,
        prepro_tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
    )
    features = read_biored("./dataset/biored/processed_test.pubtator",
                           ns.prepro_tokenizer, max_seq_length=1792,
                           max_samples=None, use_direction=False)[:1]
    queries = build_multipair_prompt(ns, features[0])

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained("/root/gpufree-data/Qwen3-8B-Base", use_fast=True)

    llm = LLM(model="/root/gpufree-data/Qwen3-8B-Base", dtype='bfloat16',
              tensor_parallel_size=1, trust_remote_code=True, enable_lora=True,
              max_model_len=2400, enable_prefix_caching=True, disable_log_stats=True,
              gpu_memory_utilization=0.80, max_num_seqs=8, enforce_eager=False)
    lora = LoRARequest("lora", 1, "results/biored_finetune/D/checkpoint")

    sp_no_lp = SamplingParams(max_tokens=512, temperature=0.0)
    sp_with_lp = SamplingParams(max_tokens=512, temperature=0.0, logprobs=20)

    # Pick first 5 source-entity queries (so we have multiple data points)
    print(f"Running {min(5, len(queries))} queries each in two modes")
    prompts = []
    for q in queries[:5]:
        chat = build_chat_prompt(q['prompt'], tok)
        ids = tok.encode(chat, add_special_tokens=False)
        prompts.append({"prompt_token_ids": ids})

    out_no_lp = llm.generate(prompts, sp_no_lp, lora_request=lora)
    out_with_lp = llm.generate(prompts, sp_with_lp, lora_request=lora)

    print(f"\n=== Comparing greedy outputs (without vs with logprobs=20) ===")
    for i, (o1, o2) in enumerate(zip(out_no_lp, out_with_lp)):
        ids1 = list(o1.outputs[0].token_ids)
        ids2 = list(o2.outputs[0].token_ids)
        match = ids1 == ids2
        print(f"\nQuery {i}: match={match}, len_no_lp={len(ids1)}, len_with_lp={len(ids2)}")
        if not match:
            # Find first divergence
            for j in range(min(len(ids1), len(ids2))):
                if ids1[j] != ids2[j]:
                    ctx = max(0, j-3)
                    print(f"  First diff at pos {j}: no_lp={ids1[j]} ({tok.decode([ids1[j]])!r}) vs with_lp={ids2[j]} ({tok.decode([ids2[j]])!r})")
                    print(f"  Context (3 tokens before): {[tok.decode([t]) for t in ids1[ctx:j]]}")
                    break

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
