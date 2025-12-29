"""Attention rollout utilities for ViT-style models."""

import torch


def attention_rollout(attn_maps):
    """Compute average attention rollout across layers.

    attn_maps: list of tensors with shape (B, heads, tokens, tokens)
    """
    attn = None
    for layer_attn in attn_maps:
        layer_attn = layer_attn.mean(dim=1)
        if attn is None:
            attn = layer_attn
        else:
            attn = attn @ layer_attn
    return attn
