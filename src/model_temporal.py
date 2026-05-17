"""Lightweight temporal model: per-segment frame embeddings + temporal conv."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _resnet18_embed(pretrained: bool) -> tuple[nn.Module, int]:
    """ResNet18 up to global average pool; output dim 512."""
    try:
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
    except TypeError:
        m = models.resnet18(pretrained=pretrained)
    embed = nn.Sequential(*list(m.children())[:-1])
    return embed, 512


class SegmentEventModel(nn.Module):
    """
    Input: (B, S, 3, H, W) stacked segment-center frames.
    Output: (B, S, num_classes) logits (multi-label per time segment).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.embed, self.embed_dim = _resnet18_embed(pretrained)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv1d(self.embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = x.shape
        z = self.embed(x.reshape(b * s, c, h, w))
        z = z.flatten(1)
        z = z.reshape(b, s, self.embed_dim).transpose(1, 2)
        z = self.temporal(z)
        logits = self.head(z)
        return logits.transpose(1, 2)
