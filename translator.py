"""Directional, head-wise KV-Lingo maps (paper §3.1–3.2 and Appendix E)."""

from pathlib import Path

import torch
from torch import nn

from qwen_adapter import PreNormCache


class CacheTranslator(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        if source.config.num_hidden_layers != target.config.num_hidden_layers:
            raise ValueError("This first translator requires equal model depth")
        if source.config.num_key_value_heads != target.config.num_key_value_heads:
            raise ValueError("Head-wise translation requires equal KV head counts")
        self.source, self.target = source, target
        shape = (target.config.num_hidden_layers, target.config.num_key_value_heads,
                 target.config.head_dim, source.config.head_dim)
        self.keys = nn.Parameter(torch.zeros(shape, device=target.device, dtype=torch.float32))
        self.values = nn.Parameter(torch.zeros_like(self.keys))

    def forward(self, features: PreNormCache) -> PreNormCache:
        self.source.validate(features)
        layers = []
        for i, (keys, values) in enumerate(features.layers):
            mapped = tuple(torch.einsum("bthi,hoi->btho", tensor.float(), weight[i]).to(self.target.dtype)
                           for tensor, weight in ((keys, self.keys), (values, self.values)))
            layers.append(mapped)
        return PreNormCache(tuple(layers), features.length, self.target.identity)

    def cache(self, features):
        return self.target.native_cache(self(features))


@torch.no_grad()
def fit_closed_form(translator, examples, *, relative_tolerance=1e-8):
    """FP64 sufficient statistics; symmetric truncated pseudo-inverse, Appendix E.

    Statistics stay on CPU; no prefix activations are retained between examples.
    There is no bias or ridge penalty in the paper's deployed initialiser.
    """
    shape = translator.keys.shape
    grams = torch.zeros((2, shape[0], shape[1], shape[3], shape[3]), dtype=torch.float64)
    crosses = torch.zeros((2, shape[0], shape[1], shape[3], shape[2]), dtype=torch.float64)
    positions = 0
    for index, example in enumerate(examples):
        tokens = torch.tensor([example["prompt_ids"][:-1]], device=translator.source.device)
        source = translator.source.capture(tokens)
        target = translator.target.capture(tokens.to(translator.target.device))
        for layer, (src_pair, tgt_pair) in enumerate(zip(source.layers, target.layers)):
            for side, (x, y) in enumerate(zip(src_pair, tgt_pair)):
                # MPS cannot hold FP64 tensors; transfer before promoting.
                x, y = x[0].cpu().double(), y[0].cpu().double()
                grams[side, layer] += torch.einsum("thi,thj->hij", x, x)
                crosses[side, layer] += torch.einsum("thi,thj->hij", x, y)
        positions += source.length
        print(f"Closed-form statistics {index + 1}/{len(examples)} ({positions} prefix positions)", flush=True)
    if not positions:
        raise ValueError("Closed-form fitting needs nonempty training examples")
    ranks = []
    for side, parameter in enumerate((translator.keys, translator.values)):
        for layer in range(shape[0]):
            values, vectors = torch.linalg.eigh(grams[side, layer])
            keep = values > values[:, -1:] * relative_tolerance
            inverse = torch.where(keep & (values > 0), values.clamp_min(torch.finfo(values.dtype).tiny).reciprocal(), 0)
            solution = (vectors * inverse.unsqueeze(-2)) @ vectors.transpose(-1, -2) @ crosses[side, layer]
            if not torch.isfinite(solution).all():
                raise RuntimeError("Nonfinite closed-form solution")
            parameter[layer].copy_(solution.transpose(-1, -2).to(parameter))
            ranks.extend(keep.sum(-1).tolist())
    return {"examples": len(examples), "prefix_positions": positions, "statistics_dtype": "float64",
            "solver": "symmetric_eigendecomposition_truncated_pseudoinverse",
            "relative_eigenvalue_tolerance": relative_tolerance, "min_rank": min(ranks), "max_rank": max(ranks)}


def save_translator(path, translator, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "representation": "pre_k_norm", "architecture": "head_wise_linear_no_bias",
               "source_identity": translator.source.identity, "target_identity": translator.target.identity,
               "weights": {"keys": translator.keys.detach().cpu(), "values": translator.values.detach().cpu()},
               "metadata": metadata}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_translator(path, source, target):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema_version") != 1 or payload.get("representation") != "pre_k_norm"
            or payload.get("architecture") != "head_wise_linear_no_bias"):
        raise ValueError("Unsupported translator artifact schema or representation")
    if payload["source_identity"] != source.identity or payload["target_identity"] != target.identity:
        raise ValueError("Translator checkpoint belongs to a different directional model pair")
    translator = CacheTranslator(source, target)
    for name in ("keys", "values"):
        tensor = payload["weights"][name]
        if tensor.shape != getattr(translator, name).shape or not torch.isfinite(tensor).all():
            raise ValueError("Invalid translator weights")
    translator.load_state_dict(payload["weights"], strict=True)
    return translator.eval(), payload["metadata"]
