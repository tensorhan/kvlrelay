"""Immutable Hugging Face checkpoints and exact shared-tokenizer verification."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime import choose_device, sha256_file
from qwen_adapter import QwenAdapter, digest, semantic_config


def tokenizer_fingerprint(tokenizer):
    backend = tokenizer.backend_tokenizer
    # Runtime padding/truncation settings are not tokenizer semantics.
    data = json.loads(backend.to_str())
    data.pop("padding", None)
    data.pop("truncation", None)
    return digest({"backend": data, "special_tokens": tokenizer.special_tokens_map,
                   "chat_template": tokenizer.chat_template})


def implementation_fingerprints():
    root = Path(__file__).parent
    names = ("benchmark.py", "runtime.py", "qwen_adapter.py", "translator.py", "model_pair.py",
             "paper_data.py", "handoff_eval.py", "train_translator.py", "ifeval.py", "recipe.py", "uv.lock")
    return {name: sha256_file(root / name) for name in names}


def load_checkpoint(name, revision, device, dtype, attention, local_files_only=False):
    local = Path(name).is_dir()
    path = Path(name).resolve() if local else Path(snapshot_download(
        name, revision=revision, local_files_only=local_files_only,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"]))
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Expected an original Transformers safetensors checkpoint")
    resolved = None if local else path.name
    provenance = {"model_id": name, "requested_revision": revision, "resolved_revision": resolved,
                  "local_weight_sha256": {p.name: sha256_file(p) for p in weights} if local else None}
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                               dtype=dtype, attn_implementation=attention).to(device)
    provenance["config_sha256"] = semantic_config(model.config)
    provenance["tokenizer_sha256"] = tokenizer_fingerprint(tokenizer)
    identity = digest({k: provenance[k] for k in ("model_id", "resolved_revision", "local_weight_sha256", "config_sha256")})
    return QwenAdapter(model, identity), tokenizer, provenance


def pair_args(parser):
    parser.add_argument("--source-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--target-revision", default="main")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)


def load_pair(args):
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    device, dtype = choose_device(args.device), getattr(torch, args.dtype)
    source, st, sp = load_checkpoint(args.source_model, args.source_revision, device, dtype, args.attention, args.local_files_only)
    target, tt, tp = load_checkpoint(args.target_model, args.target_revision, device, dtype, args.attention, args.local_files_only)
    if sp["tokenizer_sha256"] != tp["tokenizer_sha256"]:
        raise ValueError("Tokenizers/chat templates differ; cross-tokenizer translation is unsupported")
    if source.config.vocab_size != target.config.vocab_size:
        raise ValueError("Model vocabularies differ")
    if source.identity == target.identity:
        raise ValueError("A cross-model benchmark requires two distinct checkpoints")
    return source, target, tt, {"source": sp, "target": tp}
