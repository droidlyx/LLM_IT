import argparse
import os
import torch
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
    
    parser.add_argument("--use_direction", action="store_true")
    parser.add_argument("--use_augmented_training", action="store_true")
    parser.add_argument("--use_extra_training_datasets", action="store_true")

    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--phase", type=int, default=1,
                        help="for multi-step training/testing")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank (set automatically by torchrun)")
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


def finetune_llm(args, train_features, continue_training = False, previous_outputs = None):
    is_distributed = args.local_rank != -1
    is_main = args.is_main_process

    if is_main:
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*50}")
        print(f"Detected {num_gpus} GPU(s) for training")
        if is_distributed:
            print(f"Using DDP with {torch.distributed.get_world_size()} processes — full parallel utilization")
        print(f"{'='*50}")
    
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
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if is_distributed:
        torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # With DDP, Trainer gives each GPU per_device_train_batch_size samples.
    # No manual batch size multiplication needed.
    per_device_bs = int(args.llm_train_batch_size)
    if is_main:
        world = torch.distributed.get_world_size() if is_distributed else 1
        print(f"\nBatch config: per_device={per_device_bs}, GPUs={world}, "
              f"grad_accum={int(args.llm_gradient_accumulation_steps)}, "
              f"effective_global={per_device_bs * world * int(args.llm_gradient_accumulation_steps)}")

    # No device_map — Trainer handles device placement & DDP wrapping
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
    if is_main:
        print(f"Training dataset shuffled. Total samples: {len(train_dataset)}")

    # Output example training data for debugging
    if is_main:
        print("\n=== Training Data Examples ===")
        if len(all_queries) > 0:
            example_query = all_queries[0]
            example_prompt = _build_chat_or_text_prompt(example_query, tokenizer)
            full_text = (example_prompt + (example_query.get('output') or '')).strip()
            print(f"\nFull model input: {full_text}...")
            print(f"Expected output: {example_query.get('output', 'None')}")
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

    total_steps = max(1, (len(train_features) // (per_device_bs * int(args.llm_gradient_accumulation_steps))) * int(args.num_train_epochs))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=per_device_bs,
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
        warmup_steps = max(1, int(total_steps * 0.1)),
        ddp_find_unused_parameters=False,
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
    
    if is_main:
        if is_distributed:
            print(f"\nStarting DDP training with {torch.distributed.get_world_size()} GPUs — full parallel utilization")
        else:
            print("\nStarting single-GPU training...")
    
    trainer.train()
    if is_main:
        tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize distributed training if launched with torchrun/accelerate
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        mp.set_start_method('spawn', force=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    args.local_rank = local_rank
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.is_main_process = (local_rank in [-1, 0])
    set_seed(args)
    if args.use_extra_training_datasets:
        args.num_train_epochs = 1.0

    if args.is_main_process:
        print("use_direction: ", args.use_direction)
        print("use_augmented_training: ", args.use_augmented_training)
        print("use_extra_training_datasets: ", args.use_extra_training_datasets)
        if local_rank != -1:
            print(f"DDP enabled: {torch.distributed.get_world_size()} processes, local_rank={local_rank}")

    args.prepro_tokenizer = AutoTokenizer.from_pretrained('../base_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    temp_path = os.path.join(os.path.join(args.result_save_path, "checkpoint"))
    if args.is_main_process and not os.path.exists(temp_path):
        os.makedirs(temp_path)
    if local_rank != -1:
        torch.distributed.barrier()

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
    # train_features = train_features[:5]

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

        finetune_llm(args, train_features)
    else:
        raise ValueError("Unknown phase")
