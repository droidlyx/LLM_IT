import torch
from src.codi_latents import LearnableLatents


def test_latents_shape():
    lat = LearnableLatents(k=4, d_model=2560)
    out = lat(batch_size=3)
    assert out.shape == (3, 4, 2560)


def test_latents_requires_grad():
    lat = LearnableLatents(k=4, d_model=2560)
    assert lat.latents.requires_grad


def test_latents_init_small():
    lat = LearnableLatents(k=4, d_model=2560)
    # std should be ~0.02
    assert 0.005 < lat.latents.std().item() < 0.05
