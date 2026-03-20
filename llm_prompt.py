import argparse
import os
from openai import OpenAI
import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred, read_biored
from evaluation import to_official, official_evaluate, id2rel, biored_id2rel
import multiprocessing as mp
from tqdm import tqdm
import glob
from utils import text2data, remove_spaces, feature2text
from utils import collate_fn, load_data
import random
import time
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/biored", type=str)
    parser.add_argument("--train_file", default="bioredirect_train_dev.pubtator", type=str)
    parser.add_argument("--dev_file", default="bioredirect_test.pubtator", type=str)
    parser.add_argument("--test_file", default="bioredirect_bc8_test.pubtator", type=str)

    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--max_seq_length", default=3072, type=int)
    parser.add_argument("--use_direction", default="false", type=str)    
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def llm_batch_inference(queries, temperature = 0.0, use_tqdm = False):
    # Use multiprocessing to generate dataset in parallel
    num_processes = 1
    pool_args = [(prompt, i, temperature) for i, prompt in enumerate(queries)]
    with mp.Pool(processes=num_processes) as pool:
        if use_tqdm:
            results = list(tqdm(pool.imap(api_generate_wrapper, pool_args), total=len(queries)))
        else:
            results = list(pool.imap(api_generate_wrapper, pool_args))
    return results

def api_generate(prompt, run_id = 0, temperature = 0.0, stream_output = False, history=None):
    # using deepseek API
    client = OpenAI(api_key="sk-6c760b6d470d50cb5f5fd7b27d2c990e", base_url="https://apis.iflow.cn/v1")
    # Build messages: start with optional history, then system+user from current prompt
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "system", "content": prompt['instruction']})
    messages.append({"role": "user", "content": prompt['input']})

    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model='deepseek-r1',
                messages=messages,
                temperature=temperature,
                stream=stream_output
            )
            break
        except Exception as e:
            print(e)
            time.sleep(5)

    if not stream_output:
        reasoning_content = None
        if '</think>' in response.choices[0].message.content:
            reasoning_content = response.choices[0].message.content.split('</think>')[0].split('<think>')[1]
            response.choices[0].message.content = response.choices[0].message.content.split('</think>')[1]
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
        return response.choices[0].message.content, reasoning_content

    content_parts = []
    reasoning_parts = []
    for event in response:
        if not hasattr(event, 'choices') or not event.choices:
            continue
        delta = getattr(event.choices[0], 'delta', None)
        if delta is None:
            continue

        token = getattr(delta, 'content', None)
        if token:
            print(token, end='', flush=True)
            content_parts.append(token)

        token_reasoning = getattr(delta, 'reasoning_content', None)
        if token_reasoning:
            print(token_reasoning, end='', flush=True)
            reasoning_parts.append(token_reasoning)

    print('', flush=True)
    content = ''.join(content_parts)
    reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
    if '</think>' in content:
        reasoning_content = content.split('</think>')[0].split('<think>')[1]
        content = content.split('</think>')[1]

    full_content = content if reasoning_content is None else reasoning_content + content
    messages.append({"role": "assistant", "content": full_content})
    return content, reasoning_content, messages

def eval_results(args, entity_names, predicted_rels, labeled_rels):
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
            rel_name = biored_id2rel.get(rel_id, f"R{rel_id}")
        else:
            rel_name = id2rel.get(rel_id, f"R{rel_id}")
            with open('./meta/rel_info.json', 'r', encoding='utf-8') as f:
                rel_mapping = json.load(f)
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

def construct_llm_input(args, prompt, feature, labels = None):
    original_doc, entity_names = feature2text(args, feature['input_ids'][0], feature['entity_pos'][0])
    queries = []
    rel_dict = {}

    for local_idx, ht in enumerate(feature['hts'][0]):
        h_idx, t_idx = ht[0], ht[1]
        if labels is not None:
            label_rels = np.nonzero(labels[local_idx])[0].tolist()
            for rel_id in label_rels:
                if rel_id != 0:
                    if (h_idx, t_idx) not in rel_dict:
                        rel_dict[(h_idx, t_idx)] = []
                    rel_dict[(h_idx, t_idx)].append(rel_id - 1)

    for i in range(len(entity_names)):
        output_str = ''
        ent1 = f'{{{i}|{entity_names[i]}}}'
        q_cnt = 1
        phrase = f'from {ent1} to' if args.use_direction else f'between {ent1} and'
        questions = f'What is the relation {phrase} the following entities?\n'
        for j in range(len(entity_names)):
            if i!=j:
                ent2 = f'{{{j}|{entity_names[j]}}}'
                output_type = 'None'
                if (i,j) in rel_dict:
                    for rel in rel_dict[(i,j)]:
                        if output_type == 'None':
                            output_type = args.rel_list[rel]
                        else:
                            output_type += ',' + args.rel_list[rel]
                output_str += f'{q_cnt}. ' + output_type + '\n'
                questions += f'{q_cnt}.{ent2}\n'
                q_cnt += 1

        llm_input = prompt.replace('[Input Text]', original_doc)
        llm_input = llm_input.replace('[Relation list]', args.rel_list_str)
        if args.examples is not None:
            examples = '\n'.join(random.sample(args.examples, args.num_examples))
            llm_input = llm_input.replace('[Examples]', examples)
        llm_input = llm_input.replace('[Questions]', questions)
        query = {'instruction': '', 'input': llm_input, 'output': output_str if labels is not None else ''}
        queries.append(query)
        
    return queries

def evolve_instructions(args, train_features):
    dataloader = DataLoader(train_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    args.max_input_len = 0
    for batch in tqdm(dataloader, total = len(dataloader)):
        feature = ({'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    })
        labels = batch[2][0]
        queries = construct_llm_input(args, args.extract_prompt, feature)
        
        labeled_rels = {(h,t):0 for h,t in feature['hts'][0]}
        for local_idx, ht in enumerate(feature['hts'][0]):
            h_idx, t_idx = ht[0], ht[1]
            for rel_id in np.nonzero(labels[local_idx])[0].tolist():
                if rel_id != 0:
                    labeled_rels[(h_idx, t_idx)] = rel_id
                    labeled_rels[(t_idx, h_idx)] = rel_id

        with open('./meta/reflect/log.txt', 'w', buffering=1, encoding='utf-8') as flog:
            for i in range(len(queries)):
                q = queries[i]
                sq = {'instruction': '', 'input': args.summary_prompt, 'output': ''}
                with open('./meta/reflect/instructions.txt', 'r', encoding='utf-8') as f:
                    Instructions = f.read()
            
                q['input'] = q['input'].replace('[Instructions]', Instructions)
                flog.write("Input1: " + q['input'] + '\n\n')
                
                output, reasoning, history = api_generate(q, stream_output = True)
                if reasoning is not None:
                    flog.write("Reasoning1: " + reasoning + '\n\n')
                flog.write("Output1: " + output + '\n\n')

                gt = ''
                cnt = 0
                for j in range(len(queries)):
                    if i == j:
                        continue
                    cnt += 1
                    if labeled_rels[(i,j)] != 0:
                        gt += f'{cnt}. {args.rel_list[labeled_rels[(i,j)] - 1]}\n'
                    else:
                        gt += f'{cnt}. None\n'
            
                sq['input'] = sq['input'].replace('[Corrrect Answer]', gt)
                flog.write("Input2: " + sq['input'] + '\n\n')
                output_instructions, reasoning, _ = api_generate(sq, stream_output = True, history = history)

                if reasoning is not None:
                    flog.write("Reasoning2: " + reasoning + '\n\n')
                flog.write("Output2: " + output + '\n\n')

                vq = {'instruction': '', 'input': args.verify_prompt, 'output': ''}
                vq['input'] = vq['input'].replace('[Old Instruction]', Instructions)
                vq['input'] = vq['input'].replace('[New Instruction]', output_instructions)

                output, reasoning, _ = api_generate(vq, stream_output = True)

                if reasoning is not None:
                    flog.write("Reasoning3: " + reasoning + '\n\n')
                flog.write("Output3: " + output + '\n\n')

                flog.write("_____________________________________________________________\n")

                if 'yes' in output.lower():
                    with open('./meta/reflect/instructions.txt', 'w', buffering=1, encoding='utf-8') as f:
                        f.write(output_instructions)

def test_model(args, test_features, save_name = 'all_results.txt'):
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
                    })
        queries = construct_llm_input(args, args.extract_prompt, feature, labels = None)
        all_queries.extend(queries)
        all_indexes.extend([(batch_cnt,i) for i in range(len(queries))])
        if args.use_augmented_training:
            queries = construct_llm_input(args, args.extract_prompt, feature, labels = None)
            all_queries.extend(queries)
            all_indexes.extend([(batch_cnt,i) for i in range(len(queries))])
            queries = construct_llm_input(args, args.extract_prompt, feature, labels = None)
            all_queries.extend(queries)
            all_indexes.extend([(batch_cnt,i) for i in range(len(queries))])
        
    print('Total number of queries: ', len(all_queries))
    all_outputs = llm_batch_inference(all_queries, use_tqdm = True)
    for output in all_outputs[:10]:
        print(output + '\n\n')
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
                        for id, rel in enumerate(args.rel_list):
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
                if (not args.use_augmented_training) or value >= args.vote_threshold:
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

        result_string, (tp,fp,fn,f_score) = eval_results(args, entity_names, predicted_rels, labeled_rels)
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

    with open(os.path.join(args.result_save_path, save_name), 'w', encoding='utf-8') as f:
        f.write(f'final precision:{final_precision}\n')
        f.write(f'final recall:{final_recall}\n')
        f.write(f'final f1:{final_f1}\n')
        f.write(all_results)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.prepro_tokenizer = AutoTokenizer.from_pretrained('base_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    args.dataset = 'biored'
    args.use_direction = args.use_direction.lower() == 'true'

    f = open(f'./meta/reflect/extract_api.txt', 'r', encoding='utf-8')
    args.extract_prompt = f.read()
    f.close()

    f = open(f'./meta/reflect/summary_api.txt', 'r', encoding='utf-8')
    args.summary_prompt = f.read()
    f.close()

    f = open(f'./meta/reflect/verify_api.txt', 'r', encoding='utf-8')
    args.verify_prompt = f.read()
    f.close()

    f = open(f'./meta/biored_rel_list.txt', 'r', encoding='utf-8')
    args.rel_list_str = f.read()
    args.rel_list = args.rel_list_str.strip('\n').split('\n')

    train_features, dev_features, test_features = load_data(args, max_samples = None)
    evolve_instructions(args, train_features)