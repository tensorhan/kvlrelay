"""Train a directional KV-Lingo translator from the paper's Nemotron chat data.

Implements §3.2 / Appendix E's closed-form initialisation and self-distillation.
Data profiles and budgets are recorded separately from the complete paper recipe.

Usage:
  python train_translator.py --data-dir data/nemotron-pilot --output artifacts/qwen3.pt
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random

import torch
import transformers

from runtime import sha256_file
from handoff_eval import distillation_loss, evaluate
from model_pair import implementation_fingerprints, load_pair, pair_args
from paper_data import PAPER, check_disjoint, load_examples, sample_groups
from qwen_adapter import digest
from translator import CacheTranslator, fit_closed_form, load_translator, save_translator


def training_stream(examples, seed):
    """Seeded epochs that preserve the prepared mixture/reasoning schedule."""
    rng = random.Random(seed)
    groups, schedule = {}, []
    for example in examples:
        key = (example.get("kind", "natural"), example["reasoning"])
        groups.setdefault(key, []).append(example)
        schedule.append(key)
    if not schedule:
        raise ValueError("Training requires at least one example")
    while True:
        for group in groups.values():
            rng.shuffle(group)
        epoch = {key: iter(group) for key, group in groups.items()}
        for key in schedule:
            yield next(epoch[key])


def rng_state(device, saved=None):
    """Capture or restore only the generators used by this single-device run."""
    backend = getattr(torch, device.type) if device.type in ("cuda", "mps") else None
    if saved is not None:
        random.setstate(saved["python"])
        torch.set_rng_state(saved["cpu"])
        if backend is not None:
            backend.set_rng_state(saved["accelerator"], device)
        return
    return {"python": random.getstate(), "cpu": torch.get_rng_state(),
            "accelerator": backend.get_rng_state(device) if backend is not None else None}


def save_checkpoint(path, translator, optimizer, metadata, identity, history, batch_size):
    """Commit complete optimizer updates atomically; never persist a partial batch."""
    payload = {"schema_version": 1, "kind": "kvlrelay_training_checkpoint", "identity": identity,
               "weights": {name: tensor.detach().cpu() for name, tensor in translator.state_dict().items()},
               "optimizer": optimizer.state_dict(), "rng": rng_state(translator.target.device),
               "completed_steps": len(history), "samples_consumed": len(history) * batch_size,
               "history": history, "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pair_args(parser)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fit-samples", type=int, help="Required fitting count; default min(400, available conversations)")
    parser.add_argument("--distill-steps", type=int, default=100, help="Paper uses 5000; local default is a pilot")
    parser.add_argument("--batch-size", type=int, default=1, help="Gradient accumulation; paper effective batch is 8")
    parser.add_argument("--continuation-tokens", type=int, default=64, help="Paper cap is 2048")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Paper head-wise small-to-large selected rate")
    parser.add_argument("--eigenvalue-tolerance", type=float, default=1e-8)
    parser.add_argument("--init-from", type=Path, help="Reuse a compatible closed-form artifact across learning rates/seeds")
    parser.add_argument("--resume", action="store_true", help="Resume the sibling .checkpoint.pt with identical training settings")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save every N completed updates, plus step zero and the last step")
    parser.add_argument("--output", type=Path, default=Path("artifacts/translator.pt"))
    args = parser.parse_args(argv)
    if (min(args.batch_size, args.continuation_tokens, args.checkpoint_every) < 1 or args.distill_steps < 0
            or args.fit_samples is not None and args.fit_samples < 1):
        parser.error("Positive fit/batch/token counts and nonnegative distillation steps are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 < args.eigenvalue_tolerance < 1:
        parser.error("Learning rate must be positive/finite; eigenvalue tolerance must be between zero and one")
    checkpoint_path = args.output.with_suffix(".checkpoint.pt")
    baseline_path = args.output.with_name(args.output.stem + "-closed-form" + args.output.suffix)
    if args.resume and not checkpoint_path.exists():
        parser.error(f"No training checkpoint: {checkpoint_path}")
    if not args.resume and any(p.exists() for p in (args.output, checkpoint_path, baseline_path)):
        parser.error("Run artifacts already exist; use --resume or a fresh --output")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    source, target, tokenizer, provenance = load_pair(args)
    signature = provenance["target"]["tokenizer_sha256"]
    training, manifest = load_examples(args.data_dir, "train", signature, target.config.vocab_size)
    validation, _ = load_examples(args.data_dir, "validation", signature, target.config.vocab_size)
    test, _ = load_examples(args.data_dir, "test", signature, target.config.vocab_size)
    check_disjoint(training, validation, test)
    if args.fit_samples is None:
        args.fit_samples = min(400, len(training))
    if args.fit_samples > len(training):
        raise ValueError("Requested fitting count exceeds the prepared training set")
    manifest_hash = sha256_file(args.data_dir / "manifest.json")
    identity = {"models": provenance, "data_manifest_sha256": manifest_hash,
                "implementation_sha256": implementation_fingerprints(),
                "settings": {k: v for k, v in vars(args).items() if k not in
                             {"data_dir", "output", "resume", "checkpoint_every", "init_from", "local_files_only"}},
                "initialization_sha256": sha256_file(args.init_from) if args.init_from else None,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "torch": str(torch.__version__), "transformers": transformers.__version__,
                                "device": str(target.device), "cpu_threads": torch.get_num_threads()}}
    translator = CacheTranslator(source, target)
    resumed = None
    if args.resume:
        resumed = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if (resumed.get("schema_version") != 1 or resumed.get("kind") != "kvlrelay_training_checkpoint"
                or resumed.get("identity") != identity):
            raise ValueError("Training resume identity differs: data, models, code, settings or environment")
        if (not 0 <= resumed["completed_steps"] <= args.distill_steps
                or resumed["completed_steps"] != len(resumed["history"])
                or resumed["samples_consumed"] != resumed["completed_steps"] * args.batch_size
                or [h["step"] for h in resumed["history"]] != list(range(1, resumed["completed_steps"] + 1))):
            raise ValueError("Invalid checkpoint training position")
        for name, parameter in translator.state_dict().items():
            value = resumed["weights"][name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid checkpoint map weights")
        translator.load_state_dict(resumed["weights"], strict=True)
        metadata = resumed["metadata"]
        if sha256_file(baseline_path) != metadata["closed_form_sha256"]:
            raise ValueError("Closed-form baseline differs from the training checkpoint")
        fitting, baseline_validation = metadata["closed_form"], metadata["closed_form_validation"]
    elif args.init_from:
        translator, initial = load_translator(args.init_from, source, target)
        if (initial["stage"] != "closed_form" or initial["data_manifest_sha256"] != manifest_hash
                or initial["implementation_sha256"] != identity["implementation_sha256"]
                or initial["training_prompt_sha256"] != [digest(e["prompt_ids"]) for e in training]
                or initial["validation_prompt_sha256"] != [digest(e["prompt_ids"]) for e in validation]
                or initial["closed_form"]["examples"] != args.fit_samples
                or initial["closed_form"]["relative_eigenvalue_tolerance"] != args.eigenvalue_tolerance
                or initial["settings"]["model_dtype"] != args.dtype
                or initial["settings"]["attention"] != args.attention
                or initial["settings"]["continuation_cap"] != args.continuation_tokens
                or initial["environment"]["device"] != str(target.device)):
            raise ValueError("Closed-form initialization does not match this experiment")
        fitting, baseline_validation = initial["closed_form"], initial["closed_form_validation"]
    else:
        fitting = fit_closed_form(translator, training[:args.fit_samples], relative_tolerance=args.eigenvalue_tolerance)
        baseline_validation = evaluate(translator, validation, args.continuation_tokens)
    metadata = {
        "paper": PAPER, "paper_sections": ["3.1", "3.2", "Appendix E"], "full_paper_reproduction": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "models": provenance,
        "implementation_sha256": implementation_fingerprints(), "completed_distill_steps": 0,
        "data_manifest": manifest, "data_manifest_sha256": manifest_hash,
        "training_identity": identity,
        "training_groups": sorted({g for e in training for g in sample_groups(e)}),
        "validation_groups": sorted({g for e in validation for g in sample_groups(e)}),
        "training_prompt_sha256": [digest(e["prompt_ids"]) for e in training],
        "validation_prompt_sha256": [digest(e["prompt_ids"]) for e in validation],
        "closed_form": fitting, "closed_form_validation": baseline_validation,
        "settings": {"distill_steps": args.distill_steps, "effective_batch_size": args.batch_size,
                     "continuation_cap": args.continuation_tokens, "learning_rate": args.learning_rate,
                     "seed": args.seed, "model_dtype": args.dtype, "translator_dtype": "float32",
                     "sample_order": "seeded within-stratum shuffle each epoch; prepared mixture/reasoning schedule retained",
                     "optimizer": "AdamW", "weight_decay": 0.0, "gradient_clip": 1.0,
                     "schedule": "cosine with 5% warmup", "attention": args.attention},
        "environment": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                        "device": str(target.device), "cpu_threads": torch.get_num_threads()},
        "deviations": manifest.get("deviations", []) + ["independently sampled data; see manifest for actual mixture and budgets",
                       "individual training cell; learning-rate selection and seed aggregation belong to the recipe report",
                       "see settings for actual steps, effective batch, precision and continuation cap"],
    }
    if resumed is not None:
        metadata = resumed["metadata"]
    else:
        save_translator(baseline_path, translator, metadata | {"stage": "closed_form"})
        metadata["closed_form_sha256"] = sha256_file(baseline_path)
    print(f"Closed-form validation KL={baseline_validation['mean_kl_nats']:.6f}; saved {baseline_path}", flush=True)
    optimizer = torch.optim.AdamW(translator.parameters(), lr=args.learning_rate, weight_decay=0)
    warmup = max(1, math.ceil(args.distill_steps * 0.05))
    history = resumed["history"] if resumed is not None else []
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
    stream = training_stream(training, args.seed)
    for _ in range(len(history) * args.batch_size):
        next(stream)  # Replay only sample indices; no model computation or data I/O.
    if resumed is not None:
        rng_state(target.device, resumed["rng"])
        print(f"Resuming at completed update {len(history)}/{args.distill_steps}", flush=True)
    else:
        save_checkpoint(checkpoint_path, translator, optimizer, metadata, identity, history, args.batch_size)
    for step in range(len(history), args.distill_steps):
        factor = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.distill_steps - warmup)))
        optimizer.param_groups[0]["lr"] = args.learning_rate * factor
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.batch_size):
            example = next(stream)
            loss = distillation_loss(translator, example, args.continuation_tokens)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite distillation loss")
            (loss / args.batch_size).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step + 1, "mean_kl_nats": sum(losses) / len(losses),
                        "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm)})
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.distill_steps:
            save_checkpoint(checkpoint_path, translator, optimizer, metadata, identity, history, args.batch_size)
        print(f"Distillation {step + 1}/{args.distill_steps}: KL={history[-1]['mean_kl_nats']:.6f}", flush=True)
    final_validation = evaluate(translator, validation, args.continuation_tokens)
    metadata.update({"stage": "self_distilled" if args.distill_steps else "closed_form",
                     "completed_distill_steps": len(history), "training_history": history,
                     "final_validation": final_validation})
    save_translator(args.output, translator, metadata)
    report_path = args.output.with_suffix(".training.json")
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(report_path)
    print(f"Final validation KL={final_validation['mean_kl_nats']:.6f}; saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
