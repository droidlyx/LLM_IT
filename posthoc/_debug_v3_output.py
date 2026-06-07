"""Run vLLM on a single doc-0 multi-pair query and dump raw output to find drift cause."""
import os, sys, argparse, torch
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

sys.path.insert(0, '.')
sys.path.insert(0, 'posthoc')
from prepro import read_biored
from transformers import AutoTokenizer
from score_pairs_v3 import build_multipair_prompt, build_chat_prompt, find_label_positions

def main():
    ns = argparse.Namespace(
        max_input_len=0,
        extract_prompt=open("./meta/baseline/extract.txt").read(),
        use_direction=False,
        model_name_or_path="/root/gpufree-data/Qwen3-8B-Base",
        n_gpu=1, device=torch.device('cuda:0'), seed=66,
    )
    ns.prepro_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    features = read_biored("./dataset/biored/processed_test.pubtator",
                           ns.prepro_tokenizer, max_seq_length=1792,
                           max_samples=None, use_direction=False)[:1]

    queries = build_multipair_prompt(ns, features[0])
    print(f"Doc 0 query 0: h={queries[0]['h']}, {len(queries[0]['pair_jts'])} targets, expected lines 1..{len(queries[0]['pair_jts'])}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained("/root/gpufree-data/Qwen3-8B-Base", use_fast=True)
    chat = build_chat_prompt(queries[0]['prompt'], tok)
    prefix_ids = tok.encode(chat + "1. ", add_special_tokens=False)
    print(f"Prompt+1. : {len(prefix_ids)} tokens")

    llm = LLM(model="/root/gpufree-data/Qwen3-8B-Base", dtype='bfloat16',
              tensor_parallel_size=1, trust_remote_code=True, enable_lora=True,
              max_model_len=2400, enable_prefix_caching=True, disable_log_stats=True,
              gpu_memory_utilization=0.80, max_num_seqs=4, enforce_eager=False)
    lora = LoRARequest("lora", 1, "results/biored_finetune/D/checkpoint")

    sp = SamplingParams(max_tokens=512, temperature=0.0)
    out = llm.generate([{"prompt_token_ids": prefix_ids}], sp, lora_request=lora)
    gen_text = out[0].outputs[0].text
    gen_ids = list(out[0].outputs[0].token_ids)
    print(f"\n=== Generated text (len={len(gen_text)} chars, {len(gen_ids)} tokens) ===")
    print(repr(gen_text))
    print(f"\nFinish reason: {out[0].outputs[0].finish_reason}")

    positions = find_label_positions(gen_ids, tok)
    print(f"\nLine positions found: {positions}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
