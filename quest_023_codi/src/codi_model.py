"""LoRA-wrapped Qwen3-4B-Base with CODI specials + learnable latents."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from .codi_tokenizer import add_codi_specials, init_special_embeddings
from .codi_latents import LearnableLatents

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"


def load_tokenizer(path=QWEN):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_codi_trainable(path: str = QWEN, k: int = 4, lora_r: int = 16,
                         lora_alpha: int = 32, lora_dropout: float = 0.05,
                         dtype=torch.bfloat16, device: str = "cuda"):
    """Returns (peft_model, backbone, tokenizer, latents, special_ids_dict)."""
    tok = load_tokenizer(path)
    think_id, answer_id = add_codi_specials(tok)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    except (ImportError, ValueError):
        base = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="sdpa",
            trust_remote_code=True,
        )
    base.resize_token_embeddings(len(tok))
    init_special_embeddings(base, tok, think_id, answer_id)

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens"],  # let resized embed train (covers <think>/<answer>)
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora)
    backbone = model.base_model.model.model
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    model.to(device)

    d_model = base.config.hidden_size
    latents = LearnableLatents(k=k, d_model=d_model).to(device=device, dtype=dtype)

    return model, backbone, tok, latents, {"think": think_id, "answer": answer_id}
