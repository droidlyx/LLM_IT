import torch
import torch.nn as nn


class LearnableLatents(nn.Module):
    """k trainable d-dim embeddings expanded along the batch dim on call."""
    def __init__(self, k: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.latents = nn.Parameter(torch.randn(k, d_model) * init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)
