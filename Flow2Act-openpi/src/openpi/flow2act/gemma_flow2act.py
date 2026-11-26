# ruff: noqa: N801, N802, N999
# Copyright 2024 Big Vision Authors. Licensed under the Apache License, Version 2.0.
"""Gemma with a four-tap scan carry; original parameter scopes/layout are retained."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma


class Flow2ActBlock(gemma.Block):
    taps: tuple[int, ...] = (8, 11, 14, 17)

    @nn.compact
    def __call__(self, carry, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs, features, index = carry
        xs, kv_cache = super().__call__(xs, kv_cache, positions, attn_mask, adarms_cond, deterministic)
        if xs[0] is not None:
            current = xs[0][:, : features.shape[2]]
            for slot, layer in enumerate(self.taps):
                features = features.at[slot].set(jnp.where(index == layer, current, features[slot]))
        return (xs, features, index + 1), kv_cache


class Flow2ActModule(gemma.Module):
    taps: tuple[int, ...] = (8, 11, 14, 17)
    visual_tokens: int = 512

    def setup(self):
        self.embedder = gemma.Embedder(gemma.PALIGEMMA_VOCAB_SIZE, self.configs[0].width, name="embedder")
        block = nn.remat(
            Flow2ActBlock, prevent_cse=False, static_argnums=(5,), policy=jax.checkpoint_policies.nothing_saveable
        )
        self.layers = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims, taps=self.taps)
        self.final_norms = [
            gemma.RMSNorm(name="final_norm" if i == 0 else f"final_norm_{i}") for i in range(len(self.configs))
        ]

    def __call__(
        self, embedded, positions, mask, adarms_cond=None, *, kv_cache=None, deterministic=True, return_features=False
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        example = next(e for e in embedded if e is not None)
        n = min(self.visual_tokens, embedded[0].shape[1]) if embedded[0] is not None else 0
        features = jnp.zeros((len(self.taps), example.shape[0], n, self.configs[0].width), self.embed_dtype)
        (embedded, features, _), cache = self.layers(
            (embedded, features, jnp.int32(0)), kv_cache, positions, mask[:, None], adarms_cond, deterministic
        )
        output = [
            f(e, a)[0] if e is not None else None
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        return (output, cache, features) if return_features else (output, cache)
