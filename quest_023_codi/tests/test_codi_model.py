import torch
from src.codi_model import load_codi_trainable

def test_load_returns_components():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    assert "<think>" in tok.get_vocab()
    assert "<answer>" in tok.get_vocab()
    assert latents.k == 4
    assert ids["think"] >= 0 and ids["answer"] >= 0
    # LoRA params trainable + latents trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    assert latents.latents.requires_grad
