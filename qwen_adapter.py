"""Qwen3 pre-k-norm capture and differentiable target-cache construction.

Pinned to Transformers 5.18.0. Only dense, full-attention, default-RoPE Qwen3,
batch one, contiguous positions starting at zero are supported.
"""

from dataclasses import dataclass
import hashlib
import json

import torch
from transformers import DynamicCache, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def semantic_config(config) -> str:
    data = config.to_dict()
    # Location and execution backend do not change the checkpoint's architecture.
    for key in ("_name_or_path", "transformers_version", "dtype", "torch_dtype"):
        data.pop(key, None)
    return digest(data)


@dataclass(frozen=True)
class PreNormCache:
    # Each tensor: [batch, tokens, KV heads, head dimension], before k_norm/RoPE.
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    identity: str

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for pair in self.layers for t in pair)


class QwenAdapter:
    def __init__(self, model, identity: str):
        if not isinstance(model, Qwen3ForCausalLM):
            raise ValueError("Only Qwen3ForCausalLM is supported")
        if model.config.use_sliding_window or any(kind != "full_attention" for kind in model.config.layer_types):
            raise ValueError("Only full-attention Qwen3 configurations are supported")
        if model.config.rope_parameters["rope_type"] != "default":
            raise ValueError("Only default RoPE is supported; scaled/dynamic RoPE needs a separate adapter")
        self.model = model.eval().requires_grad_(False)
        self.identity = identity
        self.config = model.config

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def forward(self, tokens, *, start=0, cache=None, all_logits=False):
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("Expected a nonempty batch-one token sequence")
        if start < 0 or start + tokens.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds the configured context window")
        if (0 if cache is None else cache.get_seq_length()) != start:
            raise ValueError("Cache length must match the absolute starting position")
        return self.model(
            input_ids=tokens,
            attention_mask=torch.ones((1, start + tokens.shape[1]), device=tokens.device, dtype=torch.long),
            position_ids=torch.arange(start, start + tokens.shape[1], device=tokens.device).unsqueeze(0),
            past_key_values=cache, use_cache=True, logits_to_keep=0 if all_logits else 1,
        )

    @torch.no_grad()
    def capture(self, tokens) -> PreNormCache:
        if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("Capture needs a nonempty batch-one prefix within the context window")
        captured = [[None, None] for _ in self.model.model.layers]
        handles = []

        def hook(layer, side):
            def record(module, inputs, output):
                captured[layer][side] = output.detach().reshape(
                    1, tokens.shape[1], self.config.num_key_value_heads, self.config.head_dim)
            return record

        try:
            for i, layer in enumerate(self.model.model.layers):
                handles.append(layer.self_attn.k_proj.register_forward_hook(hook(i, 0)))
                handles.append(layer.self_attn.v_proj.register_forward_hook(hook(i, 1)))
            # Avoid the vocabulary projection, but include ordinary native-cache creation.
            self.model.model(input_ids=tokens, use_cache=True,
                             attention_mask=torch.ones_like(tokens),
                             position_ids=torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0))
        finally:
            for handle in handles:
                handle.remove()
        if any(t is None for pair in captured for t in pair):
            raise RuntimeError("A Qwen3 projection hook did not execute")
        return PreNormCache(tuple(tuple(pair) for pair in captured), tokens.shape[1], self.identity)

    def validate(self, features: PreNormCache):
        expected = (1, features.length, self.config.num_key_value_heads, self.config.head_dim)
        if features.identity != self.identity or len(features.layers) != self.config.num_hidden_layers:
            raise ValueError("Features belong to a different model")
        if not 0 < features.length <= self.config.max_position_embeddings:
            raise ValueError("Invalid feature sequence length")
        for pair in features.layers:
            if len(pair) != 2:
                raise ValueError("Each layer needs keys and values")
            for tensor in pair:
                if tensor.shape != expected or tensor.device != self.device or tensor.dtype != self.dtype:
                    raise ValueError("Feature shape, device or dtype does not match its model")

    def native_cache(self, features: PreNormCache) -> DynamicCache:
        """Keep autograd intact so target continuation loss reaches the translator."""
        self.validate(features)
        positions = torch.arange(features.length, device=self.device).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(features.layers[0][0], positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        cache = DynamicCache(config=self.config)
        for i, (raw_keys, values) in enumerate(features.layers):
            keys = self.model.model.layers[i].self_attn.k_norm(raw_keys).transpose(1, 2)
            keys = keys * cos + rotate_half(keys) * sin
            cache.update(keys.contiguous(), values.transpose(1, 2).contiguous(), i)
        return cache

    @torch.no_grad()
    def scored_attention(self, tokens, start, block=256):
        """Per layer, mean attention each key before `start` receives from queries at `start` onward.

        Averaged over all query heads; returns FP64 [layers, start] on CPU. Scores are recomputed
        from captured projections in query blocks, so no full T x T attention tensor is kept.
        """
        length, dim = tokens.shape[-1], self.config.head_dim
        if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < start < length <= self.config.max_position_embeddings:
            raise ValueError("Scored attention needs a batch-one sequence with keys before and queries from start")
        captured = [[None, None] for _ in self.model.model.layers]
        handles = []

        def hook(layer, side):
            def record(module, inputs, output):
                captured[layer][side] = output.detach().reshape(1, length, -1, dim)
            return record

        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        try:
            for i, layer in enumerate(self.model.model.layers):
                handles.append(layer.self_attn.q_proj.register_forward_hook(hook(i, 0)))
                handles.append(layer.self_attn.k_proj.register_forward_hook(hook(i, 1)))
            self.model.model(input_ids=tokens, use_cache=False, attention_mask=torch.ones_like(tokens),
                             position_ids=positions)
        finally:
            for handle in handles:
                handle.remove()
        cos, sin = self.model.model.rotary_emb(captured[0][1], positions)
        cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
        keys_index = torch.arange(length, device=tokens.device)
        weights = torch.zeros((len(captured), start), dtype=torch.float64)
        for i, (raw_queries, raw_keys) in enumerate(captured):
            attention = self.model.model.layers[i].self_attn
            queries = attention.q_norm(raw_queries).transpose(1, 2).float()
            keys = attention.k_norm(raw_keys).transpose(1, 2).float()
            queries, keys = queries * cos + rotate_half(queries) * sin, keys * cos + rotate_half(keys) * sin
            keys = keys[0].repeat_interleave(queries.shape[1] // keys.shape[1], dim=0)  # GQA, as repeat_kv
            total = torch.zeros(start, device=tokens.device)
            for first in range(start, length, block):
                scores = queries[0, :, first:first + block] @ keys.transpose(-1, -2) * attention.scaling
                rows = torch.arange(first, first + scores.shape[1], device=tokens.device)
                scores.masked_fill_(keys_index > rows[:, None], float("-inf"))
                total += scores.softmax(-1)[..., :start].sum((0, 1))
            weights[i] = total.cpu().double() / (queries.shape[1] * (length - start))
        return weights
