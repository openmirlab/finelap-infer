"""Minimal Torch layers used by the pinned EAT inference encoder.

The shape and parameter names match the author's timm Mlp. Dropout is
inactive under eval(); no timm package is needed at inference time.
"""

from collections.abc import Iterable

from torch import nn


def to_2tuple(value):
    return tuple(value) if isinstance(value, Iterable) and not isinstance(value, str) else (value, value)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)
