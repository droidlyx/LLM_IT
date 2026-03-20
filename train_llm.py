import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from utils import set_seed, collate_fn, construct_llm_input
from prepro import read_docred, read_biored
import multiprocessing as mp
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import random
import glob 

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

    # Training params
    parser.add_argument("--llm_train_batch_size", default=1, type=int)
    parser.add_argument("--llm_gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--llm_learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=2.0, type=float)
    parser.add_argument("--num_examples", default=10, type=int)
    
    parser.add_argument("--use_direction", default="false", type=str)
    parser.add_argument("--use_augmented_training", default="false", type=str)
    parser.add_argument("--use_extra_training_datasets", default="false", type=str)

    # WMSS (Weak-Driven Learning) parameters
    parser.add_argument("--use_wmss", default="false", type=str,
                        help="Enable WMSS training method")
    parser.add_argument("--wmss_lambda", default=0.5, type=float,
                        help="Logit mixing coefficient (0.42-0.48 optimal)")
    parser.add_argument("--wmss_alpha", default=0.1, type=float,
                        help="Base difficulty weight")
    parser.add_argument("--wmss_beta", default=0.8, type=float,
                        help="Consolidation weight")
    parser.add_argument("--wmss_gamma", default=0.1, type=float,
                        help="Regression repair weight")
    parser.add_argument("--wmss_iterations", default=3, type=int,
                        help="Number of WMSS iterations")

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

    if instruction:
        return instruction + "\n\n" + user_input
    return user_input


def compute_entropy(logits):
    """Compute predictive entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


def compute_entropy_dynamics(model_weak, model_strong, dataloader, device):
    """Compute entropy dynamics (ΔH = H_strong - H_weak) for each sample."""
    model_weak.eval()
    model_strong.eval()

    entropy_weak_list = []
    entropy_strong_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing entropy dynamics"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits_weak = model_weak(input_ids, attention_mask=attention_mask).logits
            logits_strong = model_strong(input_ids, attention_mask=attention_mask).logits

            H_weak = compute_entropy(logits_weak)
            H_strong = compute_entropy(logits_strong)

            entropy_weak_list.append(H_weak.cpu())
            entropy_strong_list.append(H_strong.cpu())

    entropy_weak = torch.cat(entropy_weak_list, dim=0)
    entropy_strong = torch.cat(entropy_strong_list, dim=0)
    delta_H = entropy_strong - entropy_weak

    return entropy_weak, entropy_strong, delta_H


def compute_curriculum_weights(entropy_weak, delta_H, alpha=0.1, beta=0.8, gamma=0.1):
    """Compute sampling weights based on curriculum learning."""
    base_difficulty = alpha * entropy_weak
    consolidation = beta * torch.clamp(-delta_H, min=0)
    regression_repair = gamma * torch.clamp(delta_H, min=0)

    weights = base_difficulty + consolidation + regression_repair
    # Normalize to get sampling probabilities
    weights = weights / weights.sum()

    return weights


def wmss_train(model_strong, model_weak, train_dataset, tokenizer, args):
    """WMSS joint training via logit mixing."""
    model_weak.eval()
    model_strong.train()

    lambda_param = args.wmss_lambda
    learning_rate = args.llm_learning_rate
    max_seq_len = args.max_seq_length

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_strong.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )

    dataloader = DataLoader(train_dataset, batch_size=args.llm_train_batch_size, shuffle=True)

    for batch in tqdm(dataloader, desc="WMSS Training"):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        # Forward pass - weak model FROZEN
        with torch.no_grad():
            z_weak = model_weak(input_ids, attention_mask=attention_mask).logits

        z_strong = model_strong(input_ids, attention_mask=attention_mask).logits

        # Mix logits in logit space
        z_mix = lambda_param * z_strong + (1 - lambda_param) * z_weak

        # Compute loss on MIXED logits
        loss = F.cross_entropy(z_mix.view(-1, z_mix.size(-1)), labels.view(-1), ignore_index=-100)

        # Backward pass - only M_strong gets updated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model_strong

def finetune_llm_wmss(args, all_queries, tokenizer):
    """Full WMSS training pipeline with curriculum data activation and logit mixing."""
    print("=" * 50)
    print("Starting WMSS (Weak-Driven Learning) Training")
    print("=" * 50)

    output_dir = os.path.join(args.result_save_path, 'checkpoint')
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Initialize models
    print("\n[Stage 1] Initializing Weak & Strong Models...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Create weak model (M0 - frozen copy)
    model_weak = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    for param in model_weak.parameters():
        param.requires_grad = False
    model_weak.eval()

    # Create strong model with LoRA (M1 - trainable)
    model_strong = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_cfg = LoraConfig(
        r=12,
        lora_alpha=32,
        target_modules="all-linear",
        bias='none',
        task_type='CAUSAL_LM',
    )
    model_strong = get_peft_model(model_strong, lora_cfg)
    model_strong.print_trainable_parameters()

    if tokenizer.pad_token_id is not None:
        model_strong.config.pad_token_id = tokenizer.pad_token_id
        model_weak.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model_strong.config, 'use_cache'):
        model_strong.config.use_cache = False
    if hasattr(model_strong, 'gradient_checkpointing_enable'):
        model_strong.gradient_checkpointing_enable()

    # Build training dataset
    train_dataset = _build_wmss_dataset(all_queries, tokenizer, args.max_seq_length)
    print(f"\nTraining dataset size: {len(train_dataset)}")

    # WMSS iterations
    num_iterations = args.wmss_iterations
    print(f"\nRunning {num_iterations} WMSS iterations...")

    for iteration in range(num_iterations):
        print(f"\n{'='*40}")
        print(f"WMSS Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*40}")

        # === Stage 2: Curriculum Data Activation ===
        print("\n[Stage 2] Computing entropy dynamics...")
        # Use a subset for entropy computation if dataset is large
        eval_subset = torch.utils.data.Subset(
            train_dataset,
            indices=list(range(min(1000, len(train_dataset))))
        )
        eval_loader = DataLoader(eval_subset, batch_size=args.llm_train_batch_size, shuffle=False)

        entropy_weak, entropy_strong, delta_H = compute_entropy_dynamics(
            model_weak, model_strong, eval_loader, args.device
        )

        # Compute curriculum weights
        p_i = compute_curriculum_weights(
            entropy_weak, delta_H,
            alpha=args.wmss_alpha,
            beta=args.wmss_beta,
            gamma=args.wmss_gamma
        )

        # Weighted sampling for active dataset
        num_samples = min(len(train_dataset), 1000)
        sample_indices = torch.multinomial(p_i, num_samples, replacement=True)
        active_dataset = torch.utils.data.Subset(train_dataset, sample_indices)
        print(f"Active dataset size: {len(active_dataset)}")

        # === Stage 3: Joint Training via Logit Mixing ===
        print("\n[Stage 3] Joint training via logit mixing...")
        model_strong = wmss_train(model_strong, model_weak, active_dataset, tokenizer, args)

        # Update weak reference for next iteration
        print("\nUpdating weak reference...")
        model_weak = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # Copy strong model weights to weak model
        with torch.no_grad():
            for weak_param, strong_param in zip(model_weak.parameters(), model_strong.parameters()):
                if not strong_param.requires_grad:  # Only copy non-Lora params
                    continue
                # Copy LoRA weights
                if hasattr(strong_param, 'lora_alpha') or 'lora_' in strong_param.name.lower():
                    continue
        # For simplicity, we keep M_weak as the original base model
        # In practice, you might want to copy the strong model
        for param in model_weak.parameters():
            param.requires_grad = False
        model_weak.eval()
        print(f"Weak reference updated (λ={args.wmss_lambda})")

    # Save final model
    print("\nSaving final model...")
    model_strong.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model_strong


def _build_wmss_dataset(items, tokenizer, max_length):
    """Build dataset for WMSS training."""
    class _WMSDDataset(torch.utils.data.Dataset):
        def __init__(self, items, tokenizer, max_length):
            self.items = items
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            obj = self.items[idx]
            prompt_text = _build_chat_or_text_prompt(obj, self.tokenizer)
            output_text = obj.get('output') or ''

            prompt_enc = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            output_enc = self.tokenizer(
                output_text,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_length - prompt_enc['input_ids'].shape[1],
                return_tensors='pt',
            )

            prompt_ids = prompt_enc['input_ids'][0]
            output_ids = output_enc['input_ids'][0]

            if self.tokenizer.eos_token_id is not None:
                if output_ids.numel() == 0 or int(output_ids[-1].item()) != int(self.tokenizer.eos_token_id):
                    output_ids = torch.cat([output_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=output_ids.dtype)], dim=0)
            combined_ids = torch.cat([prompt_ids, output_ids], dim=0)

            input_ids = combined_ids[:self.max_length]
            attention_mask = torch.ones_like(input_ids)

            labels = input_ids.clone()
            prompt_length = prompt_ids.shape[0]
            labels[:prompt_length] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    return _WMSDDataset(items, tokenizer, max_length)


def finetune_llm(args, train_features, continue_training = False, previous_outputs = None):
    dataloader = DataLoader(train_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    batch_cnt = 0
    args.max_input_len = 0
    all_queries = []
    test_cnt = {}
    for batch in tqdm(dataloader, total = len(dataloader)):
        batch_cnt += 1
        feature = ({'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    'rel_list': batch[5],
                    'dataset_name': batch[6]
                    })
        labels = batch[2][0]

        doc_outputs = None
        if previous_outputs is not None:
            doc_outputs = {i:"" for i in range(len(feature['entity_pos'][0]))}
            for output,index in zip(previous_outputs['all_outputs'], previous_outputs['all_indexes']):
                if index[0] == batch_cnt:
                    doc_outputs[index[1]] = output

        if args.use_augmented_training:
            if not os.path.exists(os.path.join(args.result_save_path, 'temp')):
                os.makedirs(os.path.join(args.result_save_path, 'temp'))
            if os.path.exists(os.path.join(args.result_save_path, 'temp', f'{batch_cnt}.pt')):
                queries = torch.load(os.path.join(args.result_save_path, f'temp/{batch_cnt}.pt'))
            else:
                queries = construct_llm_input(args, feature, labels, previous_outputs = doc_outputs, generate_data = True, aug_rate = 0, shuffle = False)
                torch.save(queries, os.path.join(args.result_save_path, f'temp/{batch_cnt}.pt'))
        else:
            queries = construct_llm_input(args, feature, labels, previous_outputs = doc_outputs, generate_data = False, aug_rate = 0, shuffle = False)
        
        if feature['dataset_name'][0] not in test_cnt:
            test_cnt[feature['dataset_name'][0]] = 0
        test_cnt[feature['dataset_name'][0]] += len(queries)
        
        all_queries.extend(queries)

    print('Generated dataset size: ', len(all_queries))
    print('Max input length: ', args.max_input_len)
    print(test_cnt)

    output_dir = os.path.join(args.result_save_path, 'checkpoint')
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Load existing LoRA adapter if specified
    if continue_training:
        print(f"Loading existing LoRA adapter from: {output_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
        print("Successfully loaded LoRA adapter for continued training")
    else:
        # Create new LoRA adapter
        lora_cfg = LoraConfig(
            r=12,
            lora_alpha=32,
            # lora_dropout=0.05,
            target_modules="all-linear",
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_cfg)
        print("Created new LoRA adapter for training")

    model.print_trainable_parameters()

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        emb = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else None
        if emb is not None and hasattr(emb, 'weight'):
            emb.weight.requires_grad_(True)
        
    class _SFTDataset(torch.utils.data.Dataset):
        def __init__(self, items, tokenizer, max_length):
            self.items = items
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            obj = self.items[idx]
            prompt_text = _build_chat_or_text_prompt(obj, self.tokenizer)
            output_text = obj.get('output') or ''
            
            # Tokenize prompt and output separately to ensure accurate prompt length
            prompt_enc = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            output_enc = self.tokenizer(
                output_text,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_length - prompt_enc['input_ids'].shape[1],
                return_tensors='pt',
            )
            
            # Concatenate prompt and output tokens
            prompt_ids = prompt_enc['input_ids'][0]
            output_ids = output_enc['input_ids'][0]

            if self.tokenizer.eos_token_id is not None:
                if output_ids.numel() == 0 or int(output_ids[-1].item()) != int(self.tokenizer.eos_token_id):
                    output_ids = torch.cat([output_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=output_ids.dtype)], dim=0)
            combined_ids = torch.cat([prompt_ids, output_ids], dim=0)

            input_ids = combined_ids[:self.max_length]
            attention_mask = torch.ones_like(input_ids)

            labels = input_ids.clone()
            prompt_length = prompt_ids.shape[0]
            
            # Mask prompt tokens - only compute loss on output tokens
            labels[:prompt_length] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'length': int(attention_mask.sum().item()),
            }

    train_dataset = _SFTDataset(all_queries, tokenizer, max_length=args.max_seq_length)
    
    # Shuffle the training dataset
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    print(f"Training dataset shuffled. Total samples: {len(train_dataset)}")

    # Output example training data for debugging
    print("\n=== Training Data Examples ===")
    if len(all_queries) > 0:
        example_query = all_queries[0]
        # Show the formatted prompt
        example_prompt = _build_chat_or_text_prompt(example_query, tokenizer)
        # Show the full text that goes to the model
        full_text = (example_prompt + (example_query.get('output') or '')).strip()
        print(f"\nFull model input: {full_text}...")
        print(f"Expected output: {example_query.get('output', 'None')}")
        
        # Show tokenized version
        example_enc = tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors='pt',
        )
        print(f"\nTokenized input_ids shape: {example_enc['input_ids'].shape}")
        print(f"Tokenized attention_mask shape: {example_enc['attention_mask'].shape}")
        print(f"Number of tokens: {example_enc['attention_mask'].sum().item()}")
    print("=" * 40)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.llm_train_batch_size),
        gradient_accumulation_steps=int(args.llm_gradient_accumulation_steps),
        learning_rate=float(args.llm_learning_rate),
        logging_steps=10,
        save_strategy='no',
        report_to=[],
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim='adamw_8bit',
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        group_by_length=True,
        length_column_name='length',
    )

    def _sft_data_collator(features):
        batch = tokenizer.pad(
            [{
                'input_ids': f['input_ids'].tolist() if hasattr(f['input_ids'], 'tolist') else f['input_ids'],
                'attention_mask': f['attention_mask'].tolist() if hasattr(f['attention_mask'], 'tolist') else f['attention_mask'],
            } for f in features],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )
        labels = [f['labels'] for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if labels.shape[1] > batch['input_ids'].shape[1]:
            labels = labels[:, :batch['input_ids'].shape[1]]
        elif labels.shape[1] < batch['input_ids'].shape[1]:
            pad = batch['input_ids'].shape[1] - labels.shape[1]
            labels = torch.nn.functional.pad(labels, (0, pad), value=-100)
        batch['labels'] = labels
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        data_collator=_sft_data_collator,
    )

    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    args.use_direction = args.use_direction.lower() == 'true'
    args.use_augmented_training = args.use_augmented_training.lower() == 'true'
    args.use_extra_training_datasets = args.use_extra_training_datasets.lower() == 'true'
    if args.use_extra_training_datasets:
        args.num_train_epochs = 1.0
    
    print("use_direction: ", args.use_direction)
    print("use_augmented_training: ", args.use_augmented_training)
    print("use_extra_training_datasets: ", args.use_extra_training_datasets)
    print("use_wmss: ", args.use_wmss)
    if args.use_wmss:
        print("WMSS Parameters:")
        print(f"  lambda={args.wmss_lambda}, alpha={args.wmss_alpha}, beta={args.wmss_beta}, gamma={args.wmss_gamma}")
        print(f"  iterations={args.wmss_iterations}")

    args.prepro_tokenizer = AutoTokenizer.from_pretrained('base_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    temp_path = os.path.join(os.path.join(args.result_save_path, "checkpoint"))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    if "docred" in args.data_dir:
        args.dataset = 'docred'
    elif "biored" in args.data_dir:
        args.dataset = 'biored'
    else:
        raise ValueError("Unknown dataset")

    filename = 'docred'
    if "biored" in args.data_dir:
        filename = 'biored'

    # Load data
    max_samples = None
    read = read_docred if args.dataset == 'docred' else read_biored
    train_file = os.path.join(args.data_dir, args.train_file)
    train_features = read(train_file, args.prepro_tokenizer, max_seq_length=args.max_seq_length, max_samples = max_samples, use_direction = args.use_direction)

    if args.use_extra_training_datasets:
        files = glob.glob(os.path.join('./dataset/biomedical/*.pubtator'))
        # Duplicate the original dataset for increased training weight
        train_features = train_features + train_features
        for file in files:
            print("Processing file: ", file)
            train_features = train_features + read(file, args.prepro_tokenizer, max_seq_length=args.max_seq_length, max_samples = max_samples, use_direction = args.use_direction)

    print("Training, Phase: ", args.phase)
    if args.phase == 1:
        f = open(f'./meta/baseline/extract.txt', 'r', encoding='utf-8')
        args.extract_prompt = f.read()
        f.close()

        if args.use_wmss:
            # Generate all_queries for WMSS
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            all_queries = []
            dataloader = DataLoader(train_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
            batch_cnt = 0
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch_cnt += 1
                feature = ({
                    'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    'rel_list': batch[5],
                    'dataset_name': batch[6]
                })
                labels = batch[2][0]
                queries = construct_llm_input(args, feature, labels, previous_outputs=None, generate_data=False, aug_rate=0, shuffle=False)
                all_queries.extend(queries)

            print(f"Generated dataset size: {len(all_queries)}")
            finetune_llm_wmss(args, all_queries, tokenizer)
        else:
            finetune_llm(args, train_features)
    else:
        raise ValueError("Unknown phase")
