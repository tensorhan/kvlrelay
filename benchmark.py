"""Benchmark directional KV-Lingo handoffs on the paper's Nemotron chat data.

Partial reproduction of arXiv:2609.32610v1:
  - §4.1 / Table 4: held-out continuation perplexity and target-to-handoff KL
  - §4.2: Qwen3-0.6B -> Qwen3-1.7B, head-wise linear translation
  - Appendix Y: target re-prefill versus translation plus one target step

Reports native source/target, closed-form and distilled translators. It records
deviations from the paper; short local runs do not reproduce its full task suite.
Real-data runs exit nonzero when native controls or timing stability fail.

Usage:
  python benchmark.py nemotron --data-dir data/nemotron-pilot --translator artifacts/qwen3.pt
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import random
import statistics

import torch
import transformers
from transformers import DynamicCache

from runtime import sha256_file, timed
from handoff_eval import continuation_logits, distribution_metrics, example_tensors, summarize_quality
from model_pair import implementation_fingerprints, load_pair, pair_args
from paper_data import PAPER, check_disjoint, load_examples, sample_groups
from qwen_adapter import digest
from translator import load_translator


def timing_summary(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    mean = statistics.mean(values)
    cv = statistics.stdev(values) / mean if len(values) > 1 and mean else None
    return {"samples": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "p95": p95, "coefficient_of_variation": cv}


def paired_speedup_interval(trials, path, seed, draws=2000):
    """Bootstrap paired trial blocks, preserving each trial's shared conditions.

    This interval describes this run, not independent machines or serving load.
    It cannot override the timing stability screen.
    """
    if len(trials) < 2:
        return None
    pairs = [(t["target_reprefill"]["total_ms"], t[path]["total_ms"]) for t in trials]
    if any(not math.isfinite(v) or v <= 0 for pair in pairs for v in pair):
        raise ValueError("Speedup requires positive finite durations")
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        sample = rng.choices(pairs, k=len(pairs))
        ratios.append(statistics.median(p[0] for p in sample) / statistics.median(p[1] for p in sample))
    ratios.sort()
    return {"lower": ratios[int(draws * 0.025)], "upper": ratios[min(draws - 1, int(draws * 0.975))],
            "confidence": 0.95, "resamples": draws, "method": "paired trial bootstrap, ratio of medians"}


@torch.no_grad()
def native_controls(target, prompt, labels, reference):
    """Isolate ordinary split execution from our pre-norm capture/cache adapter."""
    expected = prompt.shape[1] + labels.numel() - 1
    prefix = target.forward(prompt[:, :-1]).past_key_values
    control, control_length = continuation_logits(target, prompt, labels, prefix)
    control_metrics = distribution_metrics(reference, control, labels)
    split_logits = control.float().cpu()
    del control, prefix
    captured = target.capture(prompt[:, :-1])
    reconstructed, reconstructed_length = continuation_logits(target, prompt, labels, target.native_cache(captured))
    reconstruction_metrics = distribution_metrics(reference, reconstructed, labels)
    return {"native_split_control": control_metrics, "native_reconstruction_control": reconstruction_metrics,
            "reconstruction_vs_split_control": distribution_metrics(split_logits, reconstructed, labels),
            "native_control_lengths_valid": control_length == reconstructed_length == expected}


@torch.no_grad()
def score_example(translator, closed_form, example, steps, control_max_kl):
    source, target = translator.source, translator.target
    prompt, labels = example_tensors(example, target, steps)
    reference, length = continuation_logits(target, prompt, labels)
    reference = reference.float().cpu()
    expected = prompt.shape[1] + labels.numel() - 1
    controls = native_controls(target, prompt, labels, reference)
    features = source.capture(prompt[:, :-1].to(source.device))
    actual, translated_length = continuation_logits(target, prompt, labels, translator.cache(features))
    metrics = {"translated": distribution_metrics(reference, actual, labels)}
    del actual
    if closed_form is not None:
        baseline, baseline_length = continuation_logits(target, prompt, labels, closed_form.cache(features))
        if baseline_length != expected:
            raise RuntimeError("Closed-form cache length mismatch")
        metrics["closed_form"] = distribution_metrics(reference, baseline, labels)
        del baseline
    source_logits, source_length = continuation_logits(source, prompt.to(source.device), labels.to(source.device))
    metrics["native_source"] = distribution_metrics(reference, source_logits, labels)
    metrics["native_target"] = distribution_metrics(reference, reference, labels)
    lengths_valid = controls["native_control_lengths_valid"] and all(n == expected for n in (length, translated_length, source_length))
    return {"id": example["id"], "group_sha256": example["group_sha256"],
            "reasoning": example["reasoning"], "prompt_tokens": prompt.shape[1],
            "scored_tokens": labels.numel(), "prompt_token_sha256": digest(example["prompt_ids"]),
            "metrics": metrics, **controls,
            "cache_lengths_valid": lengths_valid,
            "control_passed": lengths_valid and all(controls[name]["mean_kl_nats"] <= control_max_kl
                for name in ("native_split_control", "native_reconstruction_control", "reconstruction_vs_split_control"))}


@torch.no_grad()
def measure_example(translator, example, repeats, warmup, seed, progress=None):
    source, target = translator.source, translator.target
    prompt, _ = example_tensors(example, target, 1)
    features = source.capture(prompt[:, :-1].to(source.device))
    native_history = target.forward(prompt[:, :-1]).past_key_values

    def handoff(features):
        cache = translator.cache(features)
        return target.forward(prompt[:, -1:], start=prompt.shape[1] - 1, cache=cache)

    def measure(operation):
        # One synchronization pair per complete path. Keep outputs and any cold
        # source features alive until the stop boundary; validate afterwards.
        (output, retained_features), elapsed = timed(operation, target.device)
        if output.past_key_values.get_seq_length() != prompt.shape[1]:
            raise RuntimeError("Timed cache length mismatch")
        return {"total_ms": elapsed}

    def existing():
        return measure(lambda: (handoff(features), None))

    def cold():
        def operation():
            new_features = source.capture(prompt[:, :-1].to(source.device))
            return handoff(new_features), new_features
        return measure(operation)

    def native():
        return measure(lambda: (target.forward(prompt), None))

    def native_cached():
        # Restore an independent fixture outside the timer: real serving already
        # has this history. Appending a token must never contaminate later trials.
        cache = DynamicCache(config=target.config)
        for index, layer in enumerate(native_history.layers):
            cache.update(layer.keys.clone(), layer.values.clone(), index)
        return measure(lambda: (target.forward(prompt[:, -1:], start=prompt.shape[1] - 1, cache=cache), None))

    operations = {"target_reprefill": native, "existing_source_cache_handoff": existing,
                  "new_prompt_handoff": cold, "existing_target_cache_continuation": native_cached}
    for _ in range(warmup):
        for operation in operations.values():
            operation()
    rng = random.Random(f"{seed}:{example['id']}")
    trials = []
    for _ in range(repeats):
        order = list(operations)
        rng.shuffle(order)
        trial = {"order": order}
        for name in order:
            gc.collect()  # Outside timing; warm source and target fixtures remain resident.
            trial[name] = operations[name]()
        trials.append(trial)
        if progress is not None:
            progress(len(trials), repeats)
    summaries = {name: timing_summary([trial[name]["total_ms"] for trial in trials]) for name in operations}
    reliable = repeats >= 20 and warmup >= 3 and all(
        summaries[name]["coefficient_of_variation"] is not None and summaries[name]["coefficient_of_variation"] <= 0.1
        for name in operations)
    diagnostics = {}
    for name in operations:
        values = [trial[name]["total_ms"] for trial in trials]
        midpoint = len(values) // 2
        diagnostics[name] = {"variation_passed": summaries[name]["coefficient_of_variation"] is not None
                             and summaries[name]["coefficient_of_variation"] <= 0.1,
                             "second_half_over_first_half_median": statistics.median(values[midpoint:]) /
                             statistics.median(values[:midpoint]) if midpoint else None}
    return {"id": example["id"], "prompt_tokens": prompt.shape[1], "timing_ms": summaries, "trials": trials,
            "path_diagnostics": diagnostics,
            "existing_cache_speedup_interval": paired_speedup_interval(trials, "existing_source_cache_handoff", seed),
            "new_prompt_speedup_interval": paired_speedup_interval(trials, "new_prompt_handoff", seed),
            "timing_stability_gate_passed": reliable,
            "stability_gate": "at least 20 repeats, 3 warmups and per-path coefficient of variation <= 10%; not a confidence guarantee",
            "median_existing_cache_speedup": summaries["target_reprefill"]["median"] / summaries["existing_source_cache_handoff"]["median"],
            "median_new_prompt_speedup": summaries["target_reprefill"]["median"] / summaries["new_prompt_handoff"]["median"],
            "source_pre_norm_cache_bytes": features.nbytes,
            "resident_native_target_history_bytes": sum(
                t.numel() * t.element_size() for layer in native_history.layers for t in (layer.keys, layer.values))}


def verification_summary(controls, timing, acceptance):
    """Separate an executed experiment from passing its initial numerical checks."""
    stable = bool(timing) and all(item["timing_stability_gate_passed"] for item in timing)
    failures = []
    if not controls:
        failures.append("native_controls")
    if not stable:
        failures.append("timing_stability")
    if acceptance is False:
        failures.append("declared_quality_limit")
    return {"scope": "paper_based_evaluation",
            "status": "checks_failed" if failures else "checks_passed",
            "failed_checks": failures, "timing_stability_passed": stable,
            "quality_assessed": acceptance is not None}


def select_timing_examples(examples, prefixes, count):
    """Choose real held-out contexts near requested lengths, without cropping."""
    if prefixes is None:
        return examples[:count]
    selected, used = [], set()
    for length in prefixes:
        candidates = [e for e in examples if e["id"] not in used
                      and abs(len(e["prompt_ids"]) - length) <= length * 0.1]
        if not candidates:
            raise ValueError(f"No unused held-out prefix within 10% of {length} tokens; prepare matching contexts")
        example = min(candidates, key=lambda e: (abs(len(e["prompt_ids"]) - length), e["id"]))
        selected.append(example)
        used.add(example["id"])
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["nemotron"])
    pair_args(parser)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--translator", type=Path, required=True)
    parser.add_argument("--closed-form", type=Path, help="Defaults to the sibling -closed-form artifact")
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-examples", type=int, default=1, help="Quality covers all test samples; timing uses this many")
    parser.add_argument("--timing-prefix-tokens", type=int, nargs="+",
                        help="Select distinct held-out prefixes within 10%% of each length; overrides --timing-examples")
    parser.add_argument("--control-max-kl", type=float, default=1e-3, help="Declared native full-vs-split numerical control limit")
    parser.add_argument("--max-kl", type=float, help="Optional application quality acceptance limit; no default paper PASS claim")
    parser.add_argument("--output", type=Path, default=Path("results/model-switch.json"))
    args = parser.parse_args(argv)
    if min(args.continuation_tokens, args.repeats, args.timing_examples) < 1 or args.warmup < 0:
        parser.error("Positive token/repeat/timing counts and nonnegative warmup are required")
    if args.timing_prefix_tokens is not None and (min(args.timing_prefix_tokens) < 2
            or len(set(args.timing_prefix_tokens)) != len(args.timing_prefix_tokens)):
        parser.error("Timing prefix lengths must be distinct and >= 2")
    if not math.isfinite(args.control_max_kl) or args.control_max_kl < 0 or args.max_kl is not None and (not math.isfinite(args.max_kl) or args.max_kl < 0):
        parser.error("KL thresholds must be finite and nonnegative")
    # Resolve the pair from immutable revisions in the artifact, rather than today's main.
    artifact_meta = torch.load(args.translator, weights_only=True, map_location="cpu")["metadata"]
    for side in ("source", "target"):
        saved = artifact_meta["models"][side]
        if getattr(args, side + "_model") == saved["model_id"] and getattr(args, side + "_revision") == "main" and saved["resolved_revision"]:
            setattr(args, side + "_revision", saved["resolved_revision"])
    source, target, tokenizer, provenance = load_pair(args)
    translator, metadata = load_translator(args.translator, source, target)
    baseline_path = args.closed_form or args.translator.with_name(args.translator.stem + "-closed-form" + args.translator.suffix)
    if "closed_form_sha256" in metadata and sha256_file(baseline_path) != metadata["closed_form_sha256"]:
        raise ValueError("Closed-form baseline checksum differs from the trained artifact")
    closed_form, baseline_meta = load_translator(baseline_path, source, target)
    if baseline_meta["stage"] != "closed_form" or baseline_meta["training_groups"] != metadata["training_groups"] or baseline_meta["data_manifest_sha256"] != metadata["data_manifest_sha256"]:
        raise ValueError("Closed-form baseline does not belong to this training run")
    signature = provenance["target"]["tokenizer_sha256"]
    examples, manifest = load_examples(args.data_dir, "test", signature, target.config.vocab_size)
    train, _ = load_examples(args.data_dir, "train", signature, target.config.vocab_size)
    validation, _ = load_examples(args.data_dir, "validation", signature, target.config.vocab_size)
    check_disjoint(train, validation, examples)
    used_groups = set(metadata["training_groups"] + metadata["validation_groups"])
    used_prompts = set(metadata["training_prompt_sha256"] + metadata["validation_prompt_sha256"])
    if any(sample_groups(e) & used_groups or digest(e["prompt_ids"]) in used_prompts for e in examples):
        raise ValueError("Evaluation overlaps the artifact's training or validation conversations")
    if sha256_file(args.data_dir / "manifest.json") != metadata["data_manifest_sha256"]:
        raise ValueError("Evaluation data manifest differs from this trained artifact")
    timing_examples = select_timing_examples(examples, args.timing_prefix_tokens, args.timing_examples)
    print(f"Model switch: {provenance['source']['model_id']} -> {provenance['target']['model_id']}", flush=True)
    observations = []
    for example in examples:
        result = score_example(translator, closed_form, example, args.continuation_tokens, args.control_max_kl)
        observations.append(result)
        print(f"{result['id']}: {result['prompt_tokens']} tokens; handoff KL={result['metrics']['translated']['mean_kl_nats']:.6f}; control={'PASS' if result['control_passed'] else 'FAIL'}", flush=True)
    quality = {name: summarize_quality([item["metrics"][name] for item in observations])
               for name in ("native_target", "native_source", "closed_form", "translated")}
    timing = []
    for index, example in enumerate(timing_examples):
        measurement = measure_example(translator, example, args.repeats, args.warmup, args.seed)
        measurement.update({"requested_prefix_tokens": args.timing_prefix_tokens[index] if args.timing_prefix_tokens else None,
                            "reasoning": example["reasoning"], "kind": example.get("kind", "natural")})
        timing.append(measurement)
        ms = measurement["timing_ms"]
        print(f"Timing {example['id']}: target re-prefill={ms['target_reprefill']['median']:.2f}ms; "
              f"existing-source handoff={ms['existing_source_cache_handoff']['median']:.2f}ms; "
              f"new-prompt handoff={ms['new_prompt_handoff']['median']:.2f}ms; "
              f"existing-target continuation={ms['existing_target_cache_continuation']['median']:.2f}ms; "
              f"stability={'PASS' if measurement['timing_stability_gate_passed'] else 'INSUFFICIENT'}", flush=True)
    controls = all(item["control_passed"] for item in observations)
    acceptance = None if args.max_kl is None else quality["translated"]["mean_kl_nats"] <= args.max_kl
    verification = verification_summary(controls, timing, acceptance)
    report = {"schema_version": 2, "benchmark": "cross_model_nemotron_continuation", "paper": PAPER,
              "paper_sections": ["3.2", "4.1", "4.2", "Table 4", "Appendix Y"],
              "full_paper_reproduction": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "models": provenance, "translator_sha256": sha256_file(args.translator),
              "implementation_sha256": implementation_fingerprints(),
              "closed_form_sha256": sha256_file(baseline_path), "training": metadata,
              "model_precision_changed_since_training": metadata["settings"]["model_dtype"] != args.dtype,
              "evaluation_data": manifest, "quality": quality, "observations": observations,
              "native_controls_passed": controls, "quality_accepted": acceptance,
              "verification": verification,
              "quality_max_kl": args.max_kl, "native_control_max_kl": args.control_max_kl,
              "timings": timing,
              "timing_protocol": "path_boundary_sync_v2: one synchronization pair per path; outputs retained through stop; no nested component timers",
              "timing_scope": "same-process, both frozen models and immutable baseline histories resident, batch one; adapter costs included; fixture cloning, loading/tokenization/metrics/network transfer excluded; first-token logits only",
              "existing_cache_scope": "source pre-k-norm history features already available; source prefill excluded",
              "existing_target_cache_scope": "valid native target history already available; fixture cloned outside timing; one final-token step; no translation",
              "new_prompt_scope": "source capture/prefill plus translation/cache construction plus target final-token step",
              "performance_claim_validated": False,
              "deviations": manifest.get("deviations", []) + metadata["deviations"] + [
                  "Nemotron continuation LM row only; no downstream task suite or paper speedup reproduction",
                  "same-stack PyTorch backend; actual hardware/precision/lengths recorded below",
                  "data preparation does not guarantee authors' sample distribution",
                  "existing native target-cache continuation is an additional engineering baseline, not a reproduced paper experiment"],
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": str(torch.__version__), "transformers": transformers.__version__,
                              "device": str(target.device), "dtype": args.dtype, "attention": args.attention,
                              "cpu_threads": torch.get_num_threads()},
              "settings": {"continuation_cap": args.continuation_tokens, "repeats": args.repeats,
                           "warmup": args.warmup, "timing_examples": len(timing), "seed": args.seed,
                           "timing_prefix_tokens": args.timing_prefix_tokens,
                           "quality_continuation": "teacher-forced dataset reply, including any retained reasoning trace"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}; initial verification={verification['status'].upper()}; quality acceptance={'UNASSESSED' if acceptance is None else 'PASS' if acceptance else 'FAIL'}", flush=True)
    return 1 if verification["failed_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
