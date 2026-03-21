import argparse
import os
from fsspec.utils import T
from openai import OpenAI
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import set_seed, collate_fn, feature2text, construct_llm_input
from prepro import read_docred, read_biored
import multiprocessing as mp
from tqdm import tqdm
from utils import remove_spaces

_LOCAL_LLM_CACHE = {}
_LOCAL_VLLM_CACHE = {}

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument("--result_save_path", default="", type=str)
    parser.add_argument("--max_seq_length", default=3072, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--model_name_or_path", default="base_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", type=str)

    # Inference params
    parser.add_argument("--llm_max_new_tokens", default=512, type=int)
    parser.add_argument("--llm_top_p", default=0.7, type=float)
    parser.add_argument("--llm_infer_batch_size", default=16, type=int)
    parser.add_argument("--use_vllm", default="true", type=str)
    parser.add_argument("--vllm_batch_size", default=256, type=int)
    parser.add_argument("--vote_threshold", default=2, type=int)
    parser.add_argument("--num_examples", default=10, type=int)

    parser.add_argument("--use_direction", action="store_true")       
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--phase", type=int, default=1,
                        help="for multi-step training/testing")
    return parser.parse_args()

def _build_chat_or_text_prompt(prompt_obj, tokenizer):
    instruction = (prompt_obj.get('instruction') or '').strip()
    user_input = (prompt_obj.get('input') or '').strip()

    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": user_input})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def llm_batch_inference(queries, temperature = 0.0, use_tqdm = False):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as e:
        raise ImportError("vLLM is not installed or LoRA support is unavailable. Please install vllm (and a version that supports LoRARequest).") from e

    finetuned_dir = args.load_dir
    lora_request = None
    if os.path.isdir(finetuned_dir) and (
        os.path.exists(os.path.join(finetuned_dir, 'adapter_config.json'))
        or os.path.exists(os.path.join(finetuned_dir, 'adapter_config.yaml'))
    ):
        lora_request = LoRARequest("lora", 1, finetuned_dir)

    cache_key = (args.model_name_or_path, finetuned_dir if lora_request is not None else None)
    if cache_key in _LOCAL_VLLM_CACHE:
        tokenizer, vllm_engine = _LOCAL_VLLM_CACHE[cache_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        vllm_engine = LLM(
            model=args.model_name_or_path,
            dtype='bfloat16',
            tensor_parallel_size=max(1, args.n_gpu),
            trust_remote_code=True,
            enable_lora=True,
            max_model_len=int(args.max_seq_length) + int(args.llm_max_new_tokens),
            enable_prefix_caching=True,
            disable_log_stats=True,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
        )
        
        _LOCAL_VLLM_CACHE[cache_key] = (tokenizer, vllm_engine)

    prompts = [_build_chat_or_text_prompt(q, tokenizer) for q in queries]
    temp = float(temperature) if temperature is not None else 0.0
    do_sample = temp > 0
    sampling_params = SamplingParams(
        max_tokens=int(args.llm_max_new_tokens),
        temperature=temp,
        top_p=float(args.llm_top_p) if do_sample else 1.0,
    )

    vllm_bs = max(1, int(args.vllm_batch_size))
    iterator = range(0, len(prompts), vllm_bs)
    if use_tqdm:
        iterator = tqdm(list(iterator))
    results = []
    for start in iterator:
        batch_prompts = prompts[start:start + vllm_bs]
        enc = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_seq_length),
        )
        vllm_inputs = [{"prompt_token_ids": ids} for ids in enc["input_ids"]]
        
        outputs = vllm_engine.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        for out in outputs:
            results.append(out.outputs[0].text)
        
    return results

def eval_results(args, entity_names, predicted_rels, labeled_rels, rel_list):
    # Convert to sets for comparison
    labeled_set = set(labeled_rels)
    predicted_set = set(predicted_rels)
    
    # Calculate correct, false positives, and false negatives
    correct = labeled_set & predicted_set
    false_positives = predicted_set - labeled_set
    false_negatives = labeled_set - predicted_set
    
    output_lines = [f"Here is your relation extraction result compared with the ground truth:\n\n"]
    
    # Helper function to format relation
    def format_relation(h_idx, t_idx, rel_id):
        import json
        h_name = entity_names.get(h_idx, f"entity_{h_idx}")
        t_name = entity_names.get(t_idx, f"entity_{t_idx}")
        
        # Convert relation ID to human-readable name
        if 'biored' in args.data_dir.lower():
            if rel_id - 1 < len(rel_list):
                rel_name = rel_list[rel_id - 1]
            else:
                rel_name = f'R{rel_id}'
        else:
            rel2id = json.load(open('dataset/docred/rel2id.json', 'r'))
            id2rel = {value: key for key, value in rel2id.items()}
            rel_name = id2rel.get(rel_id, f"R{rel_id}")
            f = open('./dataset/docred/rel_info.json', 'r')
            rel_mapping = json.load(f)
            f.close()
            rel_name = rel_mapping[rel_name]
        
        return f"  {{{h_idx}|{remove_spaces(h_name)}}} -> {rel_name} -> {{{t_idx}|{remove_spaces(t_name)}}}"
    
    # Write correct predictions
    output_lines.append(f"CORRECT ({len(correct)}):\n")
    for h_idx, t_idx, rel_id in sorted(correct):
        output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
    output_lines.append("\n")
    
    # Write false negatives (missed by model)
    output_lines.append(f"MISSED ({len(false_negatives)}):\n")
    for h_idx, t_idx, rel_id in sorted(false_negatives):
        output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
    output_lines.append("\n")
    
    # Write false positives (incorrect predictions)
    output_lines.append(f"INCORRECT ({len(false_positives)}):\n")
    for h_idx, t_idx, rel_id in sorted(false_positives):
        output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
    output_lines.append("\n")
    
    tp = len(correct)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    output_lines.append(f"F-score: {f_score:.4f}\n")

    return ''.join(output_lines), (tp,fp,fn,f_score)

def test_model(args, test_features, previous_outputs = None, eval = True, save_name = 'all_results.txt'):
    all_queries = []
    all_indexes = []
    dataloader = DataLoader(test_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    batch_cnt = 0
    for batch in tqdm(dataloader, total = len(dataloader)):
        batch_cnt += 1
        feature = ({'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    'rel_list': batch[5]
                    })
        
        doc_outputs = None
        if previous_outputs is not None:
            doc_outputs = {i:"" for i in range(len(feature['entity_pos'][0]))}
            for output,index in zip(previous_outputs['all_outputs'], previous_outputs['all_indexes']):
                if index[0] == batch_cnt:
                    doc_outputs[index[1]] = output

        queries = construct_llm_input(args, feature, labels = None, previous_outputs = doc_outputs, aug_rate = 0, shuffle = False)
        all_queries.extend(queries)
        all_indexes.extend([(batch_cnt,i) for i in range(len(queries))])
        
    print('Total number of queries: ', len(all_queries))
    all_outputs = llm_batch_inference(all_queries, use_tqdm = True)

    model_outputs = {'all_indexes': all_indexes, 'all_outputs': all_outputs}
    if not eval:
        if save_name is not None:
            torch.save(model_outputs, os.path.join(args.result_save_path, save_name))
        return model_outputs

    print(all_outputs[0] + '\n')
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_results = ""
    batch_cnt = 0
    for batch in tqdm(dataloader, total = len(dataloader)):
        batch_cnt += 1
        ## Inference
        feature = {'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    'rel_list': batch[5]
                    }
        labels = batch[2][0]
        original_doc, entity_names = feature2text(args, feature['input_ids'][0], feature['entity_pos'][0], aug_rate = 0)
        predicted_rels = []
        doc_outputs = {i:[] for i in range(len(entity_names))}
        for output,index in zip(all_outputs, all_indexes):
            if index[0] == batch_cnt:
                doc_outputs[index[1]].append(output)
        
        # Merge all predictions from difference configurations
        for i in range(len(entity_names)):
            vote = {}
            for output in doc_outputs[i]:
                output_lines = output.strip('\n').split('\n')
                pairs = []
                for j in range(len(entity_names)):
                    if i!=j:
                        pairs.append((i,j))

                for line in output_lines:
                    try:
                        line_num, pred_rel = line.split('.')
                        line_num = int(line_num)-1
                        rel_id = 0
                        for id, rel in enumerate(feature['rel_list'][0]):
                            if rel.strip().lower() in pred_rel.strip().lower():
                                rel_id = id + 1
                                break
                        if rel_id > 0:
                            if (pairs[line_num][0], pairs[line_num][1], rel_id) in vote:
                                vote[(pairs[line_num][0], pairs[line_num][1], rel_id)] += 1
                            else:
                                vote[(pairs[line_num][0], pairs[line_num][1], rel_id)] = 1
                            if not args.use_direction:
                                if (pairs[line_num][1], pairs[line_num][0], rel_id) in vote:
                                    vote[(pairs[line_num][1], pairs[line_num][0], rel_id)] += 1
                                else:
                                    vote[(pairs[line_num][1], pairs[line_num][0], rel_id)] = 1
                    except:
                        continue
                    
            for key, value in vote.items():
                predicted_rels.append(key)

        labeled_rels = []
        for local_idx, ht in enumerate(feature['hts'][0]):
            h_idx, t_idx = ht[0], ht[1]
            label_rels = np.nonzero(labels[local_idx])[0].tolist()
            for rel_id in label_rels:
                if rel_id != 0:
                    labeled_rels.append((h_idx, t_idx, rel_id))
                    if not args.use_direction:
                        labeled_rels.append((t_idx, h_idx, rel_id))

        result_string, (tp,fp,fn,f_score) = eval_results(args, entity_names, predicted_rels, labeled_rels, feature['rel_list'][0])
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_results += original_doc + '\n\n' + result_string + '\n____________________________________________________\n'

    final_precision = all_tp / (all_tp + all_fp + 1e-8)
    final_recall = all_tp / (all_tp + all_fn + 1e-8)
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
    print('final precision:', final_precision)
    print('final recall:', final_recall)
    print('final f1:', final_f1) 

    if save_name is not None:
        f = open(os.path.join(args.result_save_path, save_name), 'w')
        f.write(f'final precision:{final_precision}\n')
        f.write(f'final recall:{final_recall}\n')
        f.write(f'final f1:{final_f1}\n')
        f.write(all_results)
        f.close() 

    return model_outputs

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    args.load_dir = os.path.join(args.result_save_path, 'checkpoint')
    args.prepro_tokenizer = AutoTokenizer.from_pretrained('../base_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    args.max_input_len = 0
    if "docred" in args.data_dir:
        args.dataset = 'docred'
    elif "biored" in args.data_dir:
        args.dataset = 'biored'
    else:
        raise ValueError("Unknown dataset")

    filename = 'docred'
    if "biored" in args.data_dir:
        filename = 'biored'

    max_samples = None
    read = read_docred if args.dataset == 'docred' else read_biored
    train_file = os.path.join(args.data_dir, args.train_file)
    train_features = read(train_file, args.prepro_tokenizer, max_seq_length=args.max_seq_length, max_samples = max_samples, use_direction = args.use_direction)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_features = read(dev_file, args.prepro_tokenizer, max_seq_length=args.max_seq_length, max_samples = max_samples, use_direction = args.use_direction)
    test_file = os.path.join(args.data_dir, args.test_file)
    test_features = read(test_file, args.prepro_tokenizer, max_seq_length=args.max_seq_length, max_samples = max_samples, use_direction = args.use_direction)    

    print("Testing, Phase: ", args.phase)
    if args.phase == 1:
        f = open(f'./meta/baseline/extract.txt', 'r', encoding='utf-8')
        args.extract_prompt = f.read()
        f.close()
        dev_model_outputs = test_model(args, dev_features, eval = True, save_name = 'dev_results.txt')
        test_model_outputs = test_model(args, test_features, eval = True, save_name = 'test_results.txt')
    else:
        raise ValueError("Unknown phase")
    
