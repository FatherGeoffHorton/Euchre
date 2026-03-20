"""
models.py  —  BidNet and PlayNet
=================================
Both use the same residual MLP backbone:
  embed → N × ResBlock → head

BidNet:  input=BID_STATE_DIM,  output=3  (pass / order / loner)
PlayNet: input=PLAY_STATE_DIM, output=24 (one logit per card)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from state_encoder import BID_STATE_DIM, PLAY_STATE_DIM


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_res: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_res)])
        self.head   = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class BidNet(nn.Module):
    def __init__(self, hidden: int = 256, n_res: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = MLP(BID_STATE_DIM, hidden, 3, n_res, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlayNet(nn.Module):
    def __init__(self, hidden: int = 256, n_res: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = MLP(PLAY_STATE_DIM, hidden, 24, n_res, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infer_config(weights: dict) -> dict:
    """
    Read hidden size and n_res from checkpoint weight shapes.
    Also validates state dims match current code — raises ValueError if not.
    """
    bd = weights['bid_net']
    pd = weights['play_net']
    hidden = bd['net.embed.0.weight'].shape[0]
    n_res  = sum(1 for k in bd
                 if k.startswith('net.blocks.') and k.endswith('.net.0.weight'))
    ckpt_bid_dim  = bd['net.embed.0.weight'].shape[1]
    ckpt_play_dim = pd['net.embed.0.weight'].shape[1]
    if ckpt_bid_dim != BID_STATE_DIM:
        raise ValueError(
            f"Checkpoint BidNet input dim ({ckpt_bid_dim}) ≠ "
            f"current BID_STATE_DIM ({BID_STATE_DIM}). "
            f"Start fresh with --no-resume.")
    if ckpt_play_dim != PLAY_STATE_DIM:
        raise ValueError(
            f"Checkpoint PlayNet input dim ({ckpt_play_dim}) ≠ "
            f"current PLAY_STATE_DIM ({PLAY_STATE_DIM}). "
            f"Start fresh with --no-resume.")
    return {'hidden': hidden, 'n_res': n_res}
