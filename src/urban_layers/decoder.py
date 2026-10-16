from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fourier(xy, bands):
    scales = math.pi * 2.0 ** torch.arange(bands, device=xy.device, dtype=xy.dtype)
    angles = xy[..., None] * scales
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d, d)
        self.key_value = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def split(self, x):
        batch, length, d = x.shape
        return x.view(batch, length, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, source=None, allowed=None, cache=None):
        query = self.split(self.query(x))
        if source is None:
            key, value = self.key_value(x).chunk(2, dim=-1)
            key, value = self.split(key), self.split(value)
            if cache is not None:
                if "key" in cache:
                    key = torch.cat([cache["key"], key], dim=2)
                    value = torch.cat([cache["value"], value], dim=2)
                cache["key"], cache["value"] = key, value
            causal = query.shape[2] > 1
            out = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
        else:
            if cache is not None and "key" in cache:
                key, value = cache["key"], cache["value"]
            else:
                key, value = self.key_value(source).chunk(2, dim=-1)
                key, value = self.split(key), self.split(value)
                if cache is not None:
                    cache["key"], cache["value"] = key, value
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
        batch, _heads, length, _d = out.shape
        return self.out(out.transpose(1, 2).reshape(batch, length, -1))


class Layer(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(d)
        self.self_attention = Attention(d, heads)
        self.norm_cross = nn.LayerNorm(d)
        self.cross_attention = Attention(d, heads)
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, allowed, cache=None):
        x = x + self.dropout(self.self_attention(self.norm_self(x), cache=None if cache is None else cache["self"]))
        x = x + self.dropout(self.cross_attention(self.norm_cross(x), memory, allowed, None if cache is None else cache["cross"]))
        return x + self.dropout(self.mlp(self.norm_mlp(x)))


class TokenDecoder(nn.Module):
    def __init__(self, vocab, features, d=256, layers=6, heads=8, max_len=1024, dropout=0.0):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.token = nn.Embedding(vocab, d)
        self.register_buffer("features", features, persistent=False)
        self.feature = nn.Linear(features.shape[1], d)
        self.position = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([Layer(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def embed(self, tokens, start=0):
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device).clamp_max(self.max_len - 1)
        return self.token(tokens) + self.feature(self.features[tokens]) + self.position(positions)[None]

    def forward(self, tokens, memory, padding):
        allowed = (~padding)[:, None, None, :]
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, memory, allowed)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, memory, padding, grammar, max_len=None, temperature=1.0, top_p=0.95, generator=None, greedy=False):
        max_len = min(max_len or self.max_len, self.max_len)
        batch = memory.shape[0]
        device = memory.device
        allowed = (~padding)[:, None, None, :]
        caches = [{"self": {}, "cross": {}} for _ in self.layers]
        tokens = torch.full((batch, 1), 1, dtype=torch.long, device=device)
        state = grammar.start(batch, device)
        x = self.embed(tokens)
        for position in range(1, max_len):
            for layer, cache in zip(self.layers, caches):
                x = layer(x, memory, allowed, cache)
            logits = self.head(self.norm(x[:, -1])).float()
            valid = grammar.allowed(state)
            logits = logits.masked_fill(~valid, float("-inf"))
            if greedy:
                token = logits.argmax(dim=-1)
            else:
                token = sample(logits / max(temperature, 1e-4), top_p, generator)
            state = grammar.advance(state, token)
            tokens = torch.cat([tokens, token[:, None]], dim=1)
            if bool(grammar.done(state).all()):
                break
            x = self.embed(token[:, None], position)
        return tokens


def sample(logits, top_p, generator=None):
    probability = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probability, order = probability.sort(dim=-1, descending=True)
        cumulative = sorted_probability.cumsum(dim=-1)
        drop = cumulative - sorted_probability > top_p
        sorted_probability = sorted_probability.masked_fill(drop, 0.0)
        probability = torch.zeros_like(probability).scatter(-1, order, sorted_probability)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probability, 1, generator=generator)[:, 0]


def sequence_loss(logits, tokens):
    # predict token t+1 from tokens up to t, ignore padding
    target = tokens[:, 1:]
    logits = logits[:, :-1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), ignore_index=0)


def pad_sequences(sequences, length=None):
    length = length or max(len(item) for item in sequences)
    result = np.zeros((len(sequences), length), dtype=np.int64)
    for index, item in enumerate(sequences):
        item = item[:length]
        result[index, : len(item)] = item
    return torch.from_numpy(result)
