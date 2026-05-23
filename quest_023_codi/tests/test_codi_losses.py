import torch
import random
from src.codi_model import load_codi_trainable
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

def test_one_step_runs_no_nan():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    label_dict = {"A": ["template a one", "template a two"],
                  "B": ["template b one", "template b two"]}
    mean = torch.zeros(model.config.hidden_size, device="cuda")
    embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids,
                                              label_dict, mean, device="cuda")
    cache = CODILabelCache(label_dict, embs_all)
    batch = {
        "task_text": ["Question 1?", "Question 2?"],
        "reasoning": ["This is reasoning one ok ok ok.", "Another reasoning two."],
        "label_idx": torch.tensor([0, 1]),
    }
    cfg = CODILossConfig(k=4, tau=0.07, alpha=0.5, beta=1.0, gamma=0.5,
                          warmup_steps=200, ramp_steps=200)
    rng = random.Random(0)
    label_embs = cache.sample_per_label(rng).to("cuda")
    loss, log = codi_step(model, backbone, latents, tok, ids, batch,
                            label_embs, mean, cfg, step=0)
    assert torch.isfinite(loss)
    assert log["L_cls_S"] > 0
    # At step 0, β and γ should be 0 (warmup)
    assert log["beta_eff"] == 0.0 and log["gamma_eff"] == 0.0

def test_warmup_ramp():
    cfg = CODILossConfig(k=4, warmup_steps=200, ramp_steps=200, beta=1.0, gamma=0.5)
    from src.codi_losses import _warmup_weights
    # Before warmup: 0
    assert _warmup_weights(0, cfg) == (0.0, 0.0)
    # End of warmup, ramp start: still 0
    assert _warmup_weights(200, cfg) == (0.0, 0.0)
    # Middle of ramp: 50% of target
    b, g = _warmup_weights(300, cfg)
    assert abs(b - 0.5) < 1e-4 and abs(g - 0.25) < 1e-4
    # End of ramp: full
    b, g = _warmup_weights(400, cfg)
    assert abs(b - 1.0) < 1e-4 and abs(g - 0.5) < 1e-4
