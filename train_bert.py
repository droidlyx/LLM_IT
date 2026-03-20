import argparse
import os

import numpy as np
import torch
from torch import autocast
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred, read_biored
from evaluation import to_official, official_evaluate, id2rel, biored_id2rel
import wandb


def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epoch, num_steps, scaler):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                with autocast(device_type='cuda'):
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, args.use_offical, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        # pred = report(args, model, test_features)
                        # with open("result.json", "w") as fh:
                        #     json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path + 'best.pt')
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scaler = torch.amp.GradScaler('cuda')
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, scaler)


def evaluate(args, model, features, official = True, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    # Generate human-readable prediction file
    best_f1 = _save_human_readable_predictions(args, preds, features, tag)

    if official:
        ans = to_official(preds, features)
        if len(ans) > 0:
            best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
        }
    else:
        output = {
            tag + "_F1": best_f1 * 100
        }
    
    return best_f1, output


def _save_human_readable_predictions(args, preds, features, tag):
    """Save predictions in human-readable format."""
    # Determine if we're using PubTator format (BioRED) or JSON format (DocRED)
    data_file = os.path.join(args.data_dir, args.dev_file if tag == "dev" else args.test_file)
    is_pubtator = data_file.endswith('.pubtator') or 'biored' in args.data_dir.lower()
    
    if is_pubtator:
        # Parse PubTator format file
        title_to_data = _parse_pubtator_file(data_file)
    else:
        # Load JSON format file (DocRED)
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        title_to_data = {item['title']: item for item in raw_data}
    
    # Load relation descriptions if available
    rel_descriptions = {}
    try:
        rel_info_file = os.path.join(args.data_dir, 'rel_info.json')
        with open(rel_info_file, 'r') as f:
            rel_descriptions = json.load(f)
    except:
        pass  # Use relation codes if descriptions not available
    
    # Group predictions by document
    doc_predictions = {}
    idx = 0
    for feature in features:
        title = feature['title']
        if title not in doc_predictions:
            doc_predictions[title] = {'predicted': [], 'labeled': []}
        
        for local_idx, ht in enumerate(feature['hts']):
            h_idx, t_idx = ht[0], ht[1]
            pred_rels = np.nonzero(preds[idx])[0].tolist()
            label_rels = np.nonzero(feature['labels'][local_idx])[0].tolist() if 'labels' in feature else []
            
            for rel_id in pred_rels:
                if rel_id != 0:  # Skip 'Na' relation
                    doc_predictions[title]['predicted'].append((h_idx, t_idx, rel_id))
            
            for rel_id in label_rels:
                if rel_id != 0:
                    doc_predictions[title]['labeled'].append((h_idx, t_idx, rel_id))
            
            idx += 1
    
    # Write human-readable output
    output_lines = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for title, rels in doc_predictions.items():
        if title not in title_to_data:
            continue
        
        raw_doc = title_to_data[title]
        
        # Write document identifier
        output_lines.append(f"=== {title} ===\n")
        
        # Get document text and entity names based on format
        if is_pubtator:
            # PubTator format
            doc_text = raw_doc['title'] + ' ' + raw_doc['abstract']
            output_lines.append(f"{doc_text}\n\n")
            
            # Get entity names from entity list
            entity_names = [ent['text'] + f" ({ent['id']})" for ent in raw_doc['entities']]
        else:
            # DocRED format
            doc_text = ' '.join([' '.join(sent) for sent in raw_doc['sents']])
            output_lines.append(f"{doc_text}\n\n")
            
            # Get entity names from vertexSet
            entities = raw_doc['vertexSet']
            entity_names = [ent[0]['name'] for ent in entities]
        
        # Convert to sets for comparison
        labeled_set = set(rels['labeled'])
        predicted_set = set(rels['predicted'])
        
        # Calculate correct, false positives, and false negatives
        correct = labeled_set & predicted_set
        false_positives = predicted_set - labeled_set
        false_negatives = labeled_set - predicted_set
        
        # Helper function to format relation
        def format_relation(h_idx, t_idx, rel_id):
            rel_code = id2rel.get(rel_id, f"R{rel_id}")
            if 'biored' in args.data_dir.lower():
                rel_code = biored_id2rel.get(rel_id, f"R{rel_id}")
            rel_desc = rel_descriptions.get(rel_code, rel_code)
            return f"  {entity_names[h_idx]} -> {rel_desc} -> {entity_names[t_idx]}"
        
        # Write correct predictions
        if correct:
            output_lines.append(f"CORRECT ({len(correct)}):\n")
            for h_idx, t_idx, rel_id in sorted(correct):
                output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
            output_lines.append("\n")
        
        # Write false negatives (missed by model)
        if false_negatives:
            output_lines.append(f"MISSED ({len(false_negatives)}):\n")
            for h_idx, t_idx, rel_id in sorted(false_negatives):
                output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
            output_lines.append("\n")
        
        # Write false positives (incorrect predictions)
        if false_positives:
            output_lines.append(f"INCORRECT ({len(false_positives)}):\n")
            for h_idx, t_idx, rel_id in sorted(false_positives):
                output_lines.append(format_relation(h_idx, t_idx, rel_id) + "\n")
            output_lines.append("\n")
        
        output_lines.append("="*80 + "\n\n")

        total_tp += len(correct)
        total_fp += len(false_positives)
        total_fn += len(false_negatives)

    if args.save_path != "":
        output_file = args.save_path + f"predictions_{tag}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"Human-readable predictions saved to {output_file}")
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def _parse_pubtator_file(file_path):
    """Parse PubTator format file and return a dictionary mapping PMID to document data."""
    pmid_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        
        parts = line.split('\t')
        pmid = parts[0]
        
        # Extract PMID from title/abstract lines
        if '|t|' in pmid:
            pmid = pmid.split('|t|')[0]
        if '|a|' in pmid:
            pmid = pmid.split('|a|')[0]
        
        # Initialize PMID entry if not exists
        if pmid not in pmid_data:
            pmid_data[pmid] = {
                'title': '',
                'abstract': '',
                'entities': [],
                'entity_id_to_idx': {}
            }
        
        # Parse title line
        if '|t|' in line:
            pmid_data[pmid]['title'] = line.split('|t|')[1]
        # Parse abstract line
        elif '|a|' in line:
            pmid_data[pmid]['abstract'] = line.split('|a|')[1]
        # Parse entity annotations (6 parts: pmid, start, end, text, type, id)
        elif len(parts) == 6:
            entity_ids = parts[5].split(';')
            for entity_id in entity_ids:
                # Check if this entity ID is already in our list
                if entity_id not in pmid_data[pmid]['entity_id_to_idx']:
                    idx = len(pmid_data[pmid]['entities'])
                    pmid_data[pmid]['entity_id_to_idx'][entity_id] = idx
                    pmid_data[pmid]['entities'].append({
                        'text': parts[3],
                        'type': parts[4],
                        'id': entity_id
                    })
    
    return pmid_data


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


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

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    wandb.init(project="DocRED")

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

    if args.load_path == "":  # Training
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    # else:
    #     args.dev_file = "dev_sample.json"
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)

    # test_file = os.path.join(args.data_dir, args.test_file)
    # test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, args.use_offical, tag="dev")
        print(dev_output)
        # pred = report(args, model, test_features)
        # with open("result.json", "w") as fh:
        #     json.dump(pred, fh)


if __name__ == "__main__":
    main()
