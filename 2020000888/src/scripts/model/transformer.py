"""
anwen hu 2020/12/9
transformer decoder
revise from https://github.com/husthuaan/AoANet/models/TransformerModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention
    :param query: batch * head * N_q* d_k
    :param key: batch * head * N_k* d_k
    :param value: batch * head * N_k* d_k
    :param mask: batch * 1 * 1 * N_k
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  # batch * head * N_q * N_k
    assert len(scores.shape) == 4
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9) #
        scores = scores + mask
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn) # batch * head * N_q* N_k
    return torch.matmul(p_attn, value), p_attn


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return self.norm(output)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        # super(DecoderLayer, self).__init__()
        super().__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.memory_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Follow Figure 1 (right) for connections."""
        # m = memory
        # LayerNorm + SelfAtt + Dropout + Residual
        tgt = self.sublayer[0](tgt, lambda x: self.self_attn(x, x, x, tgt_mask))
        # LayerNorm + Att + Dropout + Residual
        tgt = self.sublayer[1](tgt, lambda x: self.memory_attn(x, memory, memory, memory_mask))
        # LayerNorm + FF + Dropout +Residual
        layer_output = self.sublayer[2](tgt, self.feed_forward)
        return layer_output


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            if len(mask.shape) == 3:  # for self attention
                mask = mask.unsqueeze(1)  # batch * N_k * N_k > batch * 1 * N_k * N_k
            else:
                mask = mask.unsqueeze(1).unsqueeze(2) # batch * 1 * 1 *  N_k
            mask = mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            mask = (1.0 - mask) * -10000.0
            assert len(mask.shape) == 4
        nbatches = query.size(0)  # batch * N_q * d_model

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # batch * head * N_q* d_k

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout) # batch * head * N_q* d_k, # batch * head * N_q* N_k

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # batch * N_q * d_model
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
