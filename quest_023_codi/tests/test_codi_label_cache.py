import torch
from src.codi_model import load_codi_trainable
from src.codi_label_cache import encode_label_templates_codi

def test_encode_shape():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    label_dict = {
        "A": ["positive case template one", "positive case template two"],
        "B": ["negative case template one", "negative case template two"],
    }
    mean = torch.zeros(model.config.hidden_size, device="cuda")
    embs = encode_label_templates_codi(backbone, tok, model, latents, ids,
                                         label_dict, mean, device="cuda")
    # (n_labels, n_templates_per_label, D)
    assert embs.shape == (2, 2, model.config.hidden_size)
    # L2 normalized after centering
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
