"""Per-token decoded view to nail down the drift."""
import os, sys, argparse, torch
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

sys.path.insert(0, '.'); sys.path.insert(0, 'posthoc')
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

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained("/root/gpufree-data/Qwen3-8B-Base", use_fast=True)
    chat = build_chat_prompt(queries[0]['prompt'], tok)
    prefix_ids = tok.encode(chat + "1. ", add_special_tokens=False)

    llm = LLM(model="/root/gpufree-data/Qwen3-8B-Base", dtype='bfloat16',
              tensor_parallel_size=1, trust_remote_code=True, enable_lora=True,
              max_model_len=2400, enable_prefix_caching=True, disable_log_stats=True,
              gpu_memory_utilization=0.80, max_num_seqs=4, enforce_eager=False)
    lora = LoRARequest("lora", 1, "results/biored_finetune/D/checkpoint")
    sp = SamplingParams(max_tokens=256, temperature=0.0)
    out = llm.generate([{"prompt_token_ids": prefix_ids}], sp, lora_request=lora)
    gen_ids = list(out[0].outputs[0].token_ids)

    print("=== Per-token dump ===")
    for i, tid in enumerate(gen_ids):
        s = tok.decode([tid])
        print(f"  {i:3} {tid:6}  {repr(s)}")
    print(f"\nTotal: {len(gen_ids)} tokens")
    print(f"Finish reason: {out[0].outputs[0].finish_reason}")

    positions = find_label_positions(gen_ids, tok)
    print(f"\nCurrent regex finds positions: {positions}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
