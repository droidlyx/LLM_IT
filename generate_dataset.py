import argparse
import os
from openai import OpenAI
import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred, read_biored
from evaluation import id2rel, biored_id2rel
import wandb
import shutil
import random
import multiprocessing as mp
from utils import text2data, remove_spaces, feature2text

# Global lock for get_bert_results to prevent concurrent CUDA operations
bert_results_lock = mp.Lock()

def generate_single(args, model, batch, run_num):
    feature = {'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'entity_pos': batch[3],
                'hts': batch[4],
                }
    
    result_string, f_score = get_bert_results(args, model, feature, batch[2])
    original_f_score = f_score
    best_f_score = f_score

    output_feature = feature.copy()
    output_feature['input_ids'] = output_feature['input_ids'].cpu()
    del output_feature['attention_mask']
    output_feature['labels'] = batch[2]
    output_data = [(result_string, output_feature, f_score)]
    text_pool = [(f_score, result_string)]

    ## get all entity names
    all_entity_names = {}
    for id in range(len(batch[3][0])):
        start = result_string.find(f'{{{id}|')
        end = start
        while result_string[end] != '}':
            end += 1
        all_entity_names[id] = result_string[start:end].split('|')[1]

    for iter in range(args.gen_num_iters):
        full_result_string = f"###Instance 1:\n" + text_pool[0][1]
        chosen = random.sample(text_pool[1:], k=np.minimum(args.gen_num_examples-1, len(text_pool)-1))
        for i in range(len(chosen)):
            full_result_string += f"###Instance {i+2}:\n" + chosen[i][1]
        
        ## generate new text using LLM
        llm_generated_text = llm_generate(iter, full_result_string)
        # Add missing entities to make sure each entity appears at least once
        for ent_id in range(len(batch[3][0])):
            if f'{{{ent_id}|' not in llm_generated_text:
                llm_generated_text += f', {{{ent_id}|' + all_entity_names[ent_id] + '}'
            
        try:
            ## convert LLM generated text to DocRED format data
            new_feature = text2data(args, batch, llm_generated_text)

            ## test BERT on the data and output results in text format
            result_string, f_score = get_bert_results(args, model, new_feature, batch[2])
        except:
            continue
        
        text_pool.append((f_score, result_string))
        text_pool.sort(key=lambda x: x[0], reverse=True)
        
        output_feature = new_feature.copy()
        output_feature['input_ids'] = output_feature['input_ids'].cpu()
        del output_feature['attention_mask']
        output_feature['labels'] = batch[2]
        output_data.append((result_string, output_feature, f_score))
        
        if f_score > best_f_score:
            best_f_score = f_score

    print(f'Document {run_num}: Original F1: {original_f_score}, Best F1: {best_f_score}')
    if not os.path.exists('./dataset/generated'):
        os.makedirs('./dataset/generated')
    torch.save(output_data, f'./dataset/generated/{run_num}.pkl')
    return
        
def generate_dataset(args, model, features):

    dataloader = DataLoader(features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    input_dataset = []
    for batch in dataloader:
        input_dataset.append(batch)

    args.gen_num_iters = 10
    args.gen_num_examples = 5

    # Use multiprocessing to generate dataset in parallel
    # You can adjust the number of processes based on your system's capabilities
    num_processes = 16
    pool = mp.Pool(processes=num_processes)

    pool_args = [(args, model, batch, i) for i, batch in enumerate(input_dataset) if not os.path.exists(f'./dataset/generated/{i}.pkl')]

    pool.starmap(generate_single, pool_args)

    pool.close()
    pool.join()
        
    return

def get_bert_results(args, model, feature, labels):
    with bert_results_lock:  # Ensure only one process can execute this block at a time
        model.eval()
        preds = []
        with torch.no_grad():
            pred, *_ = model(**feature)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)

    """Save BERT predictions in text format with labeled entities."""
    output_lines = []
    idx = 0

    entity_pos = feature['entity_pos'][0]
    hts = feature['hts'][0]
    input_ids = feature['input_ids'][0]
    labels = labels[0]
    
    labeled_text, entity_names = feature2text(args, input_ids, entity_pos)

    # Write document header
    output_lines.append(f"Here is the input article:\n{labeled_text}\n\n")
    
    # Process predictions for this document
    predicted_rels = []
    labeled_rels = []

    for local_idx, ht in enumerate(hts):
        h_idx, t_idx = ht[0], ht[1]
        pred_rels = np.nonzero(preds[idx])[0].tolist()
        label_rels = np.nonzero(labels[local_idx])[0].tolist()
        
        for rel_id in pred_rels:
            if rel_id != 0:
                predicted_rels.append((h_idx, t_idx, rel_id))
        
        for rel_id in label_rels:
            if rel_id != 0:
                labeled_rels.append((h_idx, t_idx, rel_id))
        
        idx += 1
    
    # Convert to sets for comparison
    labeled_set = set(labeled_rels)
    predicted_set = set(predicted_rels)
    
    # Calculate correct, false positives, and false negatives
    correct = labeled_set & predicted_set
    false_positives = predicted_set - labeled_set
    false_negatives = labeled_set - predicted_set
    
    output_lines.append(f"Here is the relation extraction result of the BERT model:\n\n")
    
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
            f = open('./meta/rel_info.json', 'r')
            rel_mapping = json.load(f)
            f.close()
            rel_name = rel_mapping[rel_name]
        
        return f"  {remove_spaces(h_name)} -> {rel_name} -> {remove_spaces(t_name)}"
    
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

    return ''.join(output_lines), f_score

def llm_generate(run_id, text_input):
    f = open('./meta/llm_prompt.txt', 'r', encoding='utf-8')
    prompt = f.read()
    f.close()
    input_messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_input}
        ]

    # using deepseek API
    client = OpenAI(api_key="sk-281708bfde244ea59609de4c88ee5246", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",  #'deepseek-chat', 'deepseek-reasoner'
        messages=input_messages,
        temperature=1.3, 
        stream=False
    )

    # # using local LLMs
    # client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
    # response = client.chat.completions.create(
    #     messages=input_messages, 
    #     max_tokens=8192,
    #     temperature=1.3, 
    #     model="model"
    # )
    # print(response.choices[0].message.content)

    ## Save model output and reasoning
    # reasoning_content = None
    # if '</think>' in response.choices[0].message.content:
    #     reasoning_content = response.choices[0].message.content.split('</think>')[0].split('<think>')[1]
    #     response.choices[0].message.content = response.choices[0].message.content.split('</think>')[1]
    # if hasattr(response.choices[0].message, 'reasoning_content'):
    #     reasoning_content = response.choices[0].message.reasoning_content

    # if not os.path.exists('./results/llm_reasoning'):
    #     os.makedirs('./results/llm_reasoning')
    # if not os.path.exists('./results/llm_output'):
    #     os.makedirs('./results/llm_output')

    # if reasoning_content is not None:
    #     f = open(f'./results/llm_reasoning/{run_id}.txt', 'w', encoding='utf-8')
    #     f.write(reasoning_content)
    #     f.close()

    # f = open(f'./results/llm_output/{run_id}.txt', 'w', encoding='utf-8')
    # f.write(response.choices[0].message.content)
    # f.close()

    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if "docred" in args.data_dir:
        read = read_docred
        args.use_offical = True
    elif "biored" in args.data_dir:
        read = read_biored
        args.use_offical = False
    else:
        raise ValueError("Unknown dataset")

    # train_file = os.path.join(args.data_dir, args.train_file)
    # train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
 
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)

    # test_file = os.path.join(args.data_dir, args.test_file)
    # test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager",
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    args.tokenizer = tokenizer
    model.load_state_dict(torch.load(args.load_path))

    # generate_dataset(args, model, train_features)
    generate_dataset(args, model, dev_features)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
