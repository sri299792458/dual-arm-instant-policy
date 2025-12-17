from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
from torch import nn

from .graph import HeteroGraph, EdgeType

def _segment_softmax(logits: torch.Tensor, dst_idx: torch.Tensor, num_dst: int) -> torch.Tensor:
    """Softmax over edges that share the same dst node.
    logits: (E, H)
    dst_idx: (E,)
    returns alpha: (E, H)
    """
    E, H = logits.shape
    # max per dst for numerical stability
    max_per = torch.full((num_dst, H), -1e9, device=logits.device, dtype=logits.dtype)
    idx = dst_idx[:, None].expand(E, H)
    max_per.scatter_reduce_(0, idx, logits, reduce='amax', include_self=True)
    logits = logits - max_per[dst_idx]
    exp = torch.exp(logits)
    sum_per = torch.zeros((num_dst, H), device=logits.device, dtype=logits.dtype)
    sum_per.scatter_add_(0, idx, exp)
    return exp / (sum_per[dst_idx] + 1e-9)

@dataclass
class HGTConfig:
    d_model: int = 1024
    heads: int = 16
    d_edge: int = 0
    dropout: float = 0.0

class HeteroGraphTransformerLayer(nn.Module):
    """A lightweight hetero-graph transformer layer with per-edge-type parameters and optional edge attributes.

    This is intentionally close in spirit to an HGT/hetero-attention layer:
      - Separate W_k/W_v (and edge projections) per edge type (src, rel, dst)
      - Separate W_q/W_o per node type (dst)
      - Attention normalized over incoming edges per destination node

    Works with our simple HeteroGraph container (no torch_geometric dependency).
    """

    def __init__(self, node_types: List[str], edge_types: List[EdgeType], cfg: HGTConfig):
        super().__init__()
        assert cfg.d_model % cfg.heads == 0
        self.node_types = node_types
        self.edge_types = edge_types
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.heads

        # per node type projections
        self.W_q = nn.ModuleDict({nt: nn.Linear(cfg.d_model, cfg.d_model, bias=False) for nt in node_types})
        self.W_o = nn.ModuleDict({nt: nn.Linear(cfg.d_model, cfg.d_model, bias=False) for nt in node_types})
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(cfg.d_model) for nt in node_types})

        # per edge type projections
        self.W_k = nn.ModuleDict()
        self.W_v = nn.ModuleDict()
        self.W_e_k = nn.ModuleDict()
        self.W_e_v = nn.ModuleDict()
        for (src, rel, dst) in edge_types:
            key = self._et_key((src, rel, dst))
            self.W_k[key] = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.W_v[key] = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.d_edge > 0:
                self.W_e_k[key] = nn.Linear(cfg.d_edge, cfg.d_model, bias=False)
                self.W_e_v[key] = nn.Linear(cfg.d_edge, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _et_key(et: EdgeType) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    def forward(self, g: HeteroGraph) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {nt: torch.zeros_like(g.x[nt]) for nt in self.node_types}

        for et in self.edge_types:
            if et not in g.edge_index:
                continue
            src, rel, dst = et
            idx = g.edge_index[et]
            src_idx = idx[0]
            dst_idx = idx[1]
            E = idx.shape[1]
            if E == 0:
                continue

            key = self._et_key(et)
            Xs = g.x[src]
            Xd = g.x[dst]
            # projections
            q = self.W_q[dst](Xd).view(-1, self.cfg.heads, self.d_head)           # (Ndst,H,Dh)
            k = self.W_k[key](Xs).view(-1, self.cfg.heads, self.d_head)           # (Nsrc,H,Dh)
            v = self.W_v[key](Xs).view(-1, self.cfg.heads, self.d_head)           # (Nsrc,H,Dh)

            qe = q[dst_idx]  # (E,H,Dh)
            ke = k[src_idx]
            ve = v[src_idx]

            if self.cfg.d_edge > 0 and et in g.edge_attr:
                ea = g.edge_attr[et]  # (E, d_edge)
                ke = ke + self.W_e_k[key](ea).view(E, self.cfg.heads, self.d_head)
                ve = ve + self.W_e_v[key](ea).view(E, self.cfg.heads, self.d_head)

            # attention logits
            logits = (qe * ke).sum(dim=-1) / (self.d_head ** 0.5)  # (E,H)
            alpha = _segment_softmax(logits, dst_idx, Xd.shape[0])  # (E,H)
            alpha = self.dropout(alpha)

            msg = (alpha[..., None] * ve)  # (E,H,Dh)

            # aggregate to dst
            agg = torch.zeros((Xd.shape[0], self.cfg.heads, self.d_head), device=Xd.device, dtype=Xd.dtype)
            scatter_idx = dst_idx[:, None, None].expand(E, self.cfg.heads, self.d_head)
            agg.scatter_add_(0, scatter_idx, msg)
            agg = agg.reshape(Xd.shape[0], self.cfg.d_model)

            out[dst] = out[dst] + agg

        # output projections + residual + norm
        new_x = {}
        for nt in self.node_types:
            h = self.W_o[nt](out[nt])
            h = self.dropout(h)
            new_x[nt] = self.norm[nt](g.x[nt] + h)
        return new_x

class HeteroGraphTransformer(nn.Module):
    def __init__(self, node_types: List[str], edge_types: List[EdgeType], cfg: HGTConfig, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([HeteroGraphTransformerLayer(node_types, edge_types, cfg) for _ in range(num_layers)])

    def forward(self, g: HeteroGraph) -> HeteroGraph:
        for layer in self.layers:
            g.x = layer(g)
        return g
