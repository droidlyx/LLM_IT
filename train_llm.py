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
    
    parser.add_argument("--use_direction", action="store_true")
    parser.add_argument("--use_augmented_training", action="store_true")
    parser.add_argument("--use_extra_training_datasets", action="store_true")

    # WMSS (Weak-Driven Learning) parameters
    parser.add_argument("--use_wmss", action="store_true",
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


def compute_entropy(logits):
    """Compute predictive entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


def compute_entropy_sequential(model, dataloader, device, desc="Computing entropy", show_progress=True):
    """Compute entropy for a single model."""
    model.eval()
    entropy_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=not show_progress):
            torch.cuda.set_device(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            H = compute_entropy(logits)
            # Compute mean entropy per sample (sequences have variable lengths)
            H_mean = H.mean(dim=-1)
            # Keep on GPU for distributed operations (all_gather requires GPU tensors)
            entropy_list.append(H_mean)

    return torch.cat(entropy_list, dim=0)


def compute_entropy_dynamics(model_weak, model_strong, dataloader, device):
    """Compute entropy dynamics (ΔH = H_strong - H_weak) for each sample."""
    model_weak.eval()
    model_strong.eval()
    torch.cuda.set_device(device)

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


def find_optimal_batch_size(model, sample_batch, device, initial_batch_size=1, max_batch_size=32, target_memory_util=0.8):
    """Find optimal batch size via memory probing with binary search.

    Args:
        model: The model to test with
        sample_batch: A sample batch dict with 'input_ids', 'attention_mask'
        device: CUDA device
        initial_batch_size: Starting batch size to try
        max_batch_size: Maximum batch size to consider
        target_memory_util: Target memory utilization (0.8 = 80%)

    Returns:
        Optimal batch size that fits in GPU memory
    """
    torch.cuda.empty_cache()

    def test_batch(batch_size):
        torch.cuda.empty_cache()
        try:
            # Handle batch dimension
            input_ids = sample_batch['input_ids']
            attention_mask = sample_batch['attention_mask']

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    # Binary search for max feasible batch size
    low, high = 0, max_batch_size
    while low < high:
        mid = (low + high + 1) // 2
        if test_batch(mid):
            low = mid
        else:
            high = mid - 1

    # Get memory stats at optimal level
    torch.cuda.reset_peak_memory_stats(device)
    test_batch(low if low > 0 else 1)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

    # Try to use more memory if headroom exists
    if total_memory > 0 and low > 0:
        utilization = peak_memory / total_memory if peak_memory > 0 else 0
        if utilization < target_memory_util and low < max_batch_size:
            scale_factor = (target_memory_util * total_memory) / peak_memory if peak_memory > 0 else 1
            new_batch_size = min(max_batch_size, int(low * scale_factor))
            if test_batch(new_batch_size):
                return new_batch_size

    return low if low > 0 else initial_batch_size


def wmss_train(model, train_dataset, tokenizer, args, device):
    """WMSS joint training via logit mixing using single model with adapter toggling.

    Uses model.disable_adapter() for weak-model inference, avoiding the need
    for a second full-size model and preventing GPU OOM on 24 GB cards.
    Supports DDP for true multi-GPU parallel training.
    """
    # Get base model for adapter operations (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model
    is_distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    is_main = not is_distributed or torch.distributed.get_rank() == 0
    model.train()

    lambda_param = args.wmss_lambda
    learning_rate = args.llm_learning_rate

    # Calculate training steps for this iteration
    # Strategy: Always iterate full dataset, but limit total steps to match target epochs
    epochs_per_iteration = args.num_train_epochs / args.wmss_iterations
    
    # Debug: Check trainable parameters
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"  Trainable parameter tensors: {len(trainable_params)}")
        print(f"  Total trainable parameters: {total_params:,}")
        if len(trainable_params) == 0:
            print("  WARNING: No trainable parameters found!")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
    )

    def _wmss_collate_fn(batch):
        """Collate function to handle variable-length sequences with padding."""
        max_len = max(b['input_ids'].size(0) for b in batch)
        input_ids = []
        attention_masks = []
        labels = []
        for b in batch:
            pad_len = max_len - b['input_ids'].size(0)
            input_ids.append(torch.nn.functional.pad(b['input_ids'], (0, pad_len), value=0))
            attention_masks.append(torch.nn.functional.pad(b['attention_mask'], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(b['labels'], (0, pad_len), value=-100))
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }

    # Use DistributedSampler for DDP - each GPU processes different data
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        dataloader = DataLoader(train_dataset, batch_size=args.llm_train_batch_size, sampler=sampler, collate_fn=_wmss_collate_fn)
    else:
        dataloader = DataLoader(train_dataset, batch_size=args.llm_train_batch_size, shuffle=True, collate_fn=_wmss_collate_fn)
    
    # Calculate total steps to match target epochs
    # Note: active_dataset is already curriculum-sampled, and dataloader shuffles each epoch
    total_steps = int(len(dataloader) * epochs_per_iteration)
    if is_main:
        print(f"  Training for {epochs_per_iteration:.2f} epochs ({total_steps} steps, {len(dataloader)} batches/epoch)")
        print(f"  Active dataset: {len(train_dataset)} samples, {len(dataloader)} batches")
    
    torch.cuda.set_device(device)

    total_loss = 0.0
    num_batches = 0
    step = 0
    
    # Train for target number of steps
    # Since dataloader has shuffle=True, each iteration sees random batches
    # This ensures good coverage even when training < 1 epoch
    pbar = tqdm(total=total_steps, desc=f"WMSS Training", disable=not is_main)
    while step < total_steps:
        # Reshuffle for each pass through the data
        if is_distributed:
            sampler.set_epoch(step // len(dataloader))
        
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass - weak model FROZEN (adapter disabled = base/merged weights)
            # Use detach() to prevent gradients from flowing back to weak model
            with base_model.disable_adapter():
                z_weak = model(input_ids, attention_mask=attention_mask).logits.detach()

            # Forward pass - strong model (adapter enabled automatically after context exit)
            # Model is already in train mode
            z_strong = model(input_ids, attention_mask=attention_mask).logits

            # Debug: Check if z_strong has gradients
            if step == 0 and is_main:
                print(f"  z_strong requires_grad: {z_strong.requires_grad}")
                print(f"  z_weak requires_grad: {z_weak.requires_grad}")

            # Mix logits in logit space
            # z_strong has gradients, z_weak is detached (no gradients)
            # z_mix will have gradients flowing through z_strong component
            z_mix = lambda_param * z_strong + (1 - lambda_param) * z_weak

            # Compute loss on MIXED logits
            # Gradients will flow back through z_strong (LoRA parameters)
            loss = F.cross_entropy(z_mix.view(-1, z_mix.size(-1)), labels.view(-1), ignore_index=-100)

            # Backward pass - DDP syncs gradients automatically across GPUs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            step += 1
            pbar.update(1)

            # Log every 10 steps
            if is_main and step % 10 == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
            
            # Debug: Check for zero loss issue
            if is_main and step <= 3:
                # Check label statistics
                valid_labels = labels[labels != -100]
                num_valid = valid_labels.numel()
                
                # Check prediction accuracy
                preds = z_mix.argmax(dim=-1).view(-1)
                valid_preds = preds[labels.view(-1) != -100]
                accuracy = (valid_preds == valid_labels).float().mean().item() if num_valid > 0 else 0.0
                
                # Check logits range
                logit_max = z_mix.max().item()
                logit_min = z_mix.min().item()
                
                print(f"  Step {step}: loss={loss.item():.6f}, acc={accuracy:.4f}, "
                      f"valid_tokens={num_valid}, logit_range=[{logit_min:.2f}, {logit_max:.2f}], "
                      f"z_strong.grad={z_strong.requires_grad}, z_mix.grad={z_mix.requires_grad}")

            del z_weak, z_strong, z_mix
            torch.cuda.empty_cache()
    
    pbar.close()
    
    if is_main and num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"  Training completed. {step} steps, Average loss: {avg_loss:.4f}")

    return model

def finetune_llm_wmss(args, all_queries, tokenizer):
    """Full WMSS training pipeline with curriculum data activation and logit mixing.

    Memory-efficient single-model approach: uses one model with LoRA adapter
    toggling (disable_adapter) instead of loading two separate 8B models.
    Adapter enabled = strong model, adapter disabled = weak reference.

    Supports DDP multi-GPU training via torchrun for full GPU utilization.
    Launch: torchrun --nproc_per_node=NUM_GPUS train_llm.py --use_wmss ...
    """
    is_distributed = args.local_rank != -1
    if is_distributed:
        device = torch.device(f"cuda:{args.local_rank}")
        world_size = torch.distributed.get_world_size()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
    is_main = args.is_main_process

    if is_main:
        print("=" * 50)
        print("Starting WMSS (Weak-Driven Learning) Training")
        print("=" * 50)
        if is_distributed:
            print(f"\nUsing DDP with {world_size} GPUs — each GPU loads full model, processes different data")
        else:
            print(f"\nUsing single GPU: {device}")

    output_dir = os.path.join(args.result_save_path, 'checkpoint')
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if is_distributed:
        torch.distributed.barrier()

    # Stage 1: Initialize single model with LoRA on local GPU
    if is_main:
        print("\n[Stage 1] Initializing Model with LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

    lora_cfg = LoraConfig(
        r=12,
        lora_alpha=32,
        target_modules="all-linear",
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    if is_main:
        model.print_trainable_parameters()

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    # Enable gradient checkpointing for memory efficiency (needed to avoid OOM with 2x forward pass)
    # enable_input_require_grads() is the PEFT fix to ensure gradient checkpointing works with LoRA
    model.enable_input_require_grads()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Wrap with DDP for true multi-GPU parallel training
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        base_model = model.module
        if is_main:
            print(f"Model wrapped with DDP across {world_size} GPUs")
    else:
        base_model = model

    # Build training dataset
    train_dataset = _build_wmss_dataset(all_queries, tokenizer, args.max_seq_length)
    if is_main:
        print(f"\nTraining dataset size: {len(train_dataset)}")

    # Use full training dataset for entropy computation to avoid overfitting to small subset
    # Each WMSS iteration will compute entropy on all data and sample based on curriculum weights
    eval_subset = train_dataset
    def _wmss_collate_fn(batch):
        """Collate function to handle variable-length sequences with padding."""
        max_len = max(b['input_ids'].size(0) for b in batch)
        input_ids = []
        attention_masks = []
        labels = []
        for b in batch:
            pad_len = max_len - b['input_ids'].size(0)
            input_ids.append(torch.nn.functional.pad(b['input_ids'], (0, pad_len), value=0))
            attention_masks.append(torch.nn.functional.pad(b['attention_mask'], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(b['labels'], (0, pad_len), value=-100))
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }

    # Use DistributedSampler for parallel entropy computation across GPUs
    if is_distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_subset, shuffle=False, drop_last=False
        )
        eval_loader = DataLoader(eval_subset, batch_size=args.llm_train_batch_size, sampler=eval_sampler, collate_fn=_wmss_collate_fn)
    else:
        eval_loader = DataLoader(eval_subset, batch_size=args.llm_train_batch_size, shuffle=False, collate_fn=_wmss_collate_fn)

    # WMSS iterations
    num_iterations = args.wmss_iterations
    if is_main:
        print(f"\nRunning {num_iterations} WMSS iterations...")

    for iteration in range(num_iterations):
        if is_main:
            print(f"\n{'='*40}")
            print(f"WMSS Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*40}")

        # === Stage 2: Curriculum Data Activation ===
        if is_main:
            print("\n[Stage 2] Computing entropy dynamics...")
        model.eval()

        # Parallel entropy computation: each GPU processes different data shard
        # Then gather results from all ranks
        if is_main:
            print("  Computing weak model entropy (adapter disabled) - parallel across GPUs...")
        with base_model.disable_adapter():
            local_entropy_weak = compute_entropy_sequential(model, eval_loader, device, "weak", show_progress=is_main)

        if is_main:
            print("  Computing strong model entropy (adapter enabled) - parallel across GPUs...")
        local_entropy_strong = compute_entropy_sequential(model, eval_loader, device, "strong", show_progress=is_main)

        # Gather entropy from all ranks and concatenate in correct order
        if is_distributed:
            # Gather all local entropy tensors from all ranks
            world_size = torch.distributed.get_world_size()
            
            # Prepare lists to hold gathered tensors
            gathered_weak = [torch.zeros_like(local_entropy_weak) for _ in range(world_size)]
            gathered_strong = [torch.zeros_like(local_entropy_strong) for _ in range(world_size)]
            
            # All-gather operation
            torch.distributed.all_gather(gathered_weak, local_entropy_weak)
            torch.distributed.all_gather(gathered_strong, local_entropy_strong)
            
            # Concatenate in order (each rank processed different shards)
            entropy_weak = torch.cat(gathered_weak, dim=0)[:len(eval_subset)]
            entropy_strong = torch.cat(gathered_strong, dim=0)[:len(eval_subset)]
        else:
            entropy_weak = local_entropy_weak
            entropy_strong = local_entropy_strong

        delta_H = entropy_strong - entropy_weak

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
        if is_main:
            print(f"Active dataset size: {len(active_dataset)}")

        # === Stage 3: Joint Training via Logit Mixing ===
        if is_main:
            print("\n[Stage 3] Joint training via logit mixing...")
        model = wmss_train(model, active_dataset, tokenizer, args, device)

        # Save adapter checkpoint for this iteration (only rank 0)
        iter_checkpoint_dir = os.path.join(output_dir, f'iteration_{iteration + 1}')
        if is_main:
            print(f"\nSaving iteration {iteration + 1} adapter checkpoint...")
            os.makedirs(iter_checkpoint_dir, exist_ok=True)
            save_model = model.module if is_distributed else model
            save_model.save_pretrained(iter_checkpoint_dir)
            print(f"Adapter saved to {iter_checkpoint_dir}")
        if is_distributed:
            torch.distributed.barrier()

        # Update weak reference: merge LoRA into base weights so that
        # disable_adapter() in the next iteration returns the current strong
        if iteration < num_iterations - 1:
            if is_main:
                print("Updating weak reference (merging LoRA into base)...")
            # Unwrap DDP
            if is_distributed:
                model = model.module
            model = model.merge_and_unload()
            torch.cuda.empty_cache()
            model = get_peft_model(model, lora_cfg)
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
            # Critical: enable_input_require_grads() needed for gradient checkpointing
            model.enable_input_require_grads()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            # Re-wrap with DDP
            if is_distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=False,
                )
                base_model = model.module
                torch.distributed.barrier()
            if is_main:
                print(f"Weak reference updated (λ={args.wmss_lambda})")

    # Training complete - adapters saved for each iteration
    if is_main:
        print("\n" + "="*50)
        print("WMSS Training Complete")
        print("="*50)
        tokenizer.save_pretrained(output_dir)
        print(f"\nAdapter checkpoints saved:")
        for iter_idx in range(num_iterations):
            print(f"  - {output_dir}/iteration_{iter_idx + 1}/")
        print(f"\nTo use the trained model, load adapters sequentially in test_llm.py:")
        print(f"  1. Load base model: {args.model_name_or_path}")
        print(f"  2. Load iteration_1 adapter and merge")
        print(f"  3. Load iteration_2 adapter and merge")
        print(f"  4. Load iteration_3 adapter and merge")
    
    if is_distributed:
        torch.distributed.barrier()

    return model


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
        print("use_wmss: ", args.use_wmss)
        if args.use_wmss:
            print("WMSS Parameters:")
            print(f"  lambda={args.wmss_lambda}, alpha={args.wmss_alpha}, beta={args.wmss_beta}, gamma={args.wmss_gamma}")
            print(f"  iterations={args.wmss_iterations}")
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
