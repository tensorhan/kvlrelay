"""Evaluate native Qwen3 models or cache handoffs with the official IFEval strict grader."""

import argparse
from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import sys
from urllib.request import urlopen

import numpy as np
import torch
import transformers

from model_pair import implementation_fingerprints, load_pair, pair_args
from qwen_adapter import digest
from runtime import sha256_file
from translator import load_translator

IFEVAL_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
UPSTREAM = "https://raw.githubusercontent.com/google-research/google-research/"
FILES = ("evaluation_lib.py", "instructions.py", "instructions_registry.py",
         "instructions_util.py", "data/input_data.jsonl")
NATIVE_CONDITIONS = ("native_source", "native_target")
CONDITIONS = (*NATIVE_CONDITIONS, "closed_form", "translated")


def prepare_ifeval(directory):
    """Keep the unmodified upstream grader/data and license outside source control."""
    import nltk
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "instruction_following_eval"
    sources = {}
    for name in FILES:
        relative = "instruction_following_eval/" + name
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{UPSTREAM}{IFEVAL_REVISION}/{relative}", timeout=60) as response:
            path.write_bytes(response.read())
        sources[relative] = sha256_file(path)
    package.joinpath("__init__.py").write_text("")
    sources["instruction_following_eval/__init__.py"] = sha256_file(package / "__init__.py")
    with urlopen(f"{UPSTREAM}{IFEVAL_REVISION}/LICENSE", timeout=60) as response:
        (directory / "LICENSE").write_bytes(response.read())
    sources["LICENSE"] = sha256_file(directory / "LICENSE")
    nltk.download("punkt_tab", download_dir=str(directory / "nltk_data"), raise_on_error=True)
    for path in sorted((directory / "nltk_data").rglob("*")):
        if path.is_file():
            sources[str(path.relative_to(directory))] = sha256_file(path)
    manifest = {"revision": IFEVAL_REVISION, "upstream": UPSTREAM, "files": sources}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared official IFEval at {directory}", flush=True)


def load_ifeval(directory, limit=None):
    """Validate the cached revision and contents before importing the official grader."""
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"instruction_following_eval/" + name for name in FILES} | {
        "instruction_following_eval/__init__.py", "LICENSE"}
    if manifest["revision"] != IFEVAL_REVISION or not required <= manifest["files"].keys():
        raise ValueError("IFEval revision or file manifest differs; run --prepare")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f"IFEval file checksum mismatch: {relative}")
    sys.path.insert(0, str(directory.resolve()))
    official = importlib.import_module("instruction_following_eval.evaluation_lib")
    if not Path(official.__file__).resolve().is_relative_to(directory.resolve()):
        raise ValueError("A different IFEval package is already imported")
    import nltk
    nltk.data.path.insert(0, str((directory / "nltk_data").resolve()))
    cases = official.read_prompt_list(str(directory / "instruction_following_eval/data/input_data.jsonl"))
    if len(cases) != 541:
        raise ValueError("The pinned original IFEval dataset must have 541 prompts")
    keys, prompts = set(), set()
    for case in cases:
        if case.key in keys or case.prompt in prompts or not case.prompt.strip():
            raise ValueError("IFEval keys/prompts must be unique and nonempty")
        if not case.instruction_id_list or len(case.instruction_id_list) != len(case.kwargs):
            raise ValueError("IFEval constraints and arguments differ")
        keys.add(case.key)
        prompts.add(case.prompt)
    # A stable prefix of a hash ordering; chosen before observing any model answers.
    cases.sort(key=lambda case: digest(case.key))
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError("Limit must be between 1 and 541")
        cases = cases[:limit]
    return cases, official, manifest


def grade_response(official, case, text, seed):
    """Preserve official grading, including grading capped outputs as-is."""
    from langdetect import DetectorFactory
    DetectorFactory.seed = seed
    random.seed(seed)
    result = official.test_instruction_following_strict(case, {case.prompt: text})
    return {"prompt_passed": bool(result.follow_all_instructions),
            "instructions_passed": [bool(value) for value in result.follow_instruction_list]}


def paired_interval(differences, seed):
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "method": "paired-prompt bootstrap, 2000 draws"}


def summarize_tasks(observations, max_loss=None, seed=42, *, conditions=CONDITIONS):
    complete = [o for o in observations if set(o["answers"]) == set(conditions)]
    if not complete:
        return None
    scores = {}
    for name in conditions:
        answers = [o["answers"][name] for o in complete]
        instructions = [value for answer in answers for value in answer["instructions_passed"]]
        scores[name] = {"prompt_accuracy": sum(a["prompt_passed"] for a in answers) / len(answers),
                        "instruction_accuracy": sum(instructions) / len(instructions),
                        "prompts_passed": sum(a["prompt_passed"] for a in answers),
                        "instructions_scored": len(instructions),
                        "token_cap_reached": sum(not a["eos_reached"] for a in answers)}
    comparisons = {}
    candidate = "translated" if "translated" in conditions else "native_target"
    for baseline in (name for name in conditions if name != candidate):
        differences = [int(o["answers"][candidate]["prompt_passed"]) -
                       int(o["answers"][baseline]["prompt_passed"]) for o in complete]
        comparisons[baseline] = {
            "accuracy_delta": sum(differences) / len(differences),
            "paired_95_percent_interval": paired_interval(differences, seed),
            "regressions": [o["key"] for o, d in zip(complete, differences) if d < 0],
            "gains": [o["key"] for o, d in zip(complete, differences) if d > 0]}
    observed = None
    if candidate == "translated" and max_loss is not None:
        loss = -comparisons["native_target"]["accuracy_delta"]
        observed = scores["native_target"]["prompts_passed"] > 0 and loss <= max_loss
    return {"prompts_scored": len(complete), "scores": scores, candidate + "_vs": comparisons,
            "max_accuracy_loss": max_loss, "observed_budget_met": observed,
            "population_noninferiority_established": False,
            "interpretation": "paired fixed-benchmark estimates; bootstrap interval is not a deployment guarantee"}


@torch.no_grad()
def greedy_ids(adapter, prompt, max_tokens, eos_id, cache=None):
    if max_tokens < 1 or prompt.shape[1] + max_tokens - 1 > adapter.config.max_position_embeddings:
        raise ValueError("Generation budget exceeds the context window or is nonpositive")
    eos_ids = set(eos_id if isinstance(eos_id, (tuple, list, set)) else [eos_id])
    start = 0 if cache is None else prompt.shape[1] - 1
    output = adapter.forward(prompt if cache is None else prompt[:, -1:], start=start, cache=cache)
    ids = []
    for index in range(max_tokens):
        token = int(output.logits[:, -1, :].argmax(-1).item())
        if token in eos_ids:
            return ids, True
        ids.append(token)
        if index + 1 < max_tokens:
            output = adapter.forward(torch.tensor([[token]], device=adapter.device),
                                     start=prompt.shape[1] + index, cache=output.past_key_values)
    return ids, False


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def open_report(path, identity, resume):
    if path.exists():
        if not resume:
            raise ValueError("Report already exists; use --resume or another --output")
        report = json.loads(path.read_text())
        if report.get("identity") != identity:
            raise ValueError("Resume identity differs (code, settings, model/artifact or data)")
        keys = [o["key"] for o in report["observations"]]
        if len(keys) != len(set(keys)) or not set(keys) <= set(identity["case_keys"]):
            raise ValueError("Resume contains duplicate or unexpected case keys")
        return report
    if resume:
        raise ValueError("Resume report does not exist")
    benchmark = "ifeval_strict_native" if identity.get("settings", {}).get("native_only") else "ifeval_strict_cache_handoff"
    return {"schema_version": 3, "benchmark": benchmark, "status": "incomplete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
            "paper_protocol": "Table 4: reasoning off, greedy, 1024-token budget, strict prompt accuracy",
            "full_paper_reproduction": False, "performance_claim_validated": False, "observations": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pair_args(parser)
    parser.add_argument("dataset", choices=["ifeval"])
    parser.add_argument("--prepare", action="store_true", help="Download the pinned official grader/data and NLTK resource")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ifeval"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--translator", type=Path)
    mode.add_argument("--native-only", action="store_true", help="Score source and target without translator artifacts")
    parser.add_argument("--closed-form", type=Path)
    parser.add_argument("--limit", type=int, help="Fixed hash-selected engineering subset; default all 541")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-accuracy-loss", type=float, help="Optional observed accuracy budget; 0.01 is one percentage point")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/ifeval.json"))
    args = parser.parse_args(argv)
    if args.prepare:
        prepare_ifeval(args.data_dir)
        return 0
    if not (args.translator or args.native_only) or args.max_new_tokens < 1 or args.cpu_threads < 1:
        parser.error("Use pretrained models, --translator or --native-only, and positive token/thread budgets")
    if args.native_only and (args.closed_form is not None or args.max_accuracy_loss is not None):
        parser.error("--closed-form and --max-accuracy-loss apply only to translator evaluation")
    if args.max_accuracy_loss is not None and (not math.isfinite(args.max_accuracy_loss) or not 0 <= args.max_accuracy_loss <= 1):
        parser.error("Accuracy loss must be a finite fraction between zero and one")
    cases, official, manifest = load_ifeval(args.data_dir, args.limit)
    conditions = NATIVE_CONDITIONS if args.native_only else CONDITIONS
    if args.native_only:
        # Resolve actual weights/tokenizers before accepting a resume, including mutable revisions.
        source, target, tokenizer, provenance = load_pair(args)
        artifact_identity = {"native_models": provenance,
                             "native_devices": {"source": str(source.device), "target": str(target.device)}}
    else:
        metadata = torch.load(args.translator, map_location="cpu", weights_only=True)["metadata"]
        for side in ("source", "target"):
            saved = metadata["models"][side]
            if getattr(args, side + "_model") == saved["model_id"] and getattr(args, side + "_revision") == "main":
                setattr(args, side + "_revision", saved["resolved_revision"])
        baseline = args.closed_form or args.translator.with_name(args.translator.stem + "-closed-form" + args.translator.suffix)
        if "closed_form_sha256" in metadata and sha256_file(baseline) != metadata["closed_form_sha256"]:
            raise ValueError("Closed-form baseline checksum differs from the trained artifact")
        artifact_identity = {"translator_sha256": sha256_file(args.translator), "closed_form_sha256": sha256_file(baseline)}
    identity = {"case_keys": [c.key for c in cases], "dataset_manifest": manifest,
                **artifact_identity,
                "implementation_sha256": implementation_fingerprints() | {Path(__file__).name: sha256_file(Path(__file__))},
                "settings": {k: v for k, v in vars(args).items() if k not in
                             {"prepare", "resume", "output", "data_dir", "translator", "closed_form"}},
                "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                "torch": str(torch.__version__), "transformers": transformers.__version__,
                                "grader_dependencies": {name: version(name) for name in
                                    ("absl-py", "immutabledict", "langdetect", "nltk", "numpy")}}}
    report = open_report(args.output, identity, args.resume)
    report["scope"] = "engineering subset" if len(cases) < 541 else "original 541-prompt IFEval suite"
    report["protocol_deviations"] = (["fixed engineering subset"] if len(cases) < 541 else []) + (
        ["generation budget differs from paper"] if args.max_new_tokens != 1024 else []) + [
        "recorded chat-template wrapper, not an exact-paper numerical reproduction"] + (
        [] if args.native_only else ["one trained artifact"])
    if report["status"] == "complete":
        print(f"Already complete: {args.output}", flush=True)
        return 1 if report["task_quality"]["observed_budget_met"] is False else 0
    used = set()
    if not args.native_only:
        source, target, tokenizer, provenance = load_pair(args)
        translator, metadata = load_translator(args.translator, source, target)
        closed_form, baseline_metadata = load_translator(baseline, source, target)
        if (baseline_metadata["stage"] != "closed_form" or
                baseline_metadata["training_groups"] != metadata["training_groups"] or
                baseline_metadata["data_manifest_sha256"] != metadata["data_manifest_sha256"]):
            raise ValueError("Closed-form baseline does not belong to this training run")
        report["training"] = metadata
        report["model_precision_changed_since_training"] = metadata["settings"]["model_dtype"] != args.dtype
        used = set(metadata["training_prompt_sha256"] + metadata["validation_prompt_sha256"])
    report["models"] = provenance
    save_report(args.output, report)
    observations = {o["key"]: o for o in report["observations"]}
    for case in cases:
        observation = observations.get(case.key)
        if observation is not None and set(observation["answers"]) == set(conditions):
            continue
        text = tokenizer.apply_chat_template([{"role": "user", "content": case.prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if digest(prompt_ids) in used:
            raise ValueError("IFEval prompt overlaps training/validation")
        if len(prompt_ids) < 2 or len(prompt_ids) > 2048:
            raise ValueError(f"IFEval {case.key}: prompt outside paper's 2048-token cap; no silent truncation")
        if observation is None:
            observation = {"key": case.key, "prompt": case.prompt, "prompt_tokens": len(prompt_ids),
                           "instruction_id_list": case.instruction_id_list, "kwargs": case.kwargs, "answers": {}}
            report["observations"].append(observation)
        prompt = torch.tensor([prompt_ids], device=target.device)
        features = None if args.native_only else source.capture(prompt[:, :-1].to(source.device))
        for name in conditions:
            if name in observation["answers"]:
                continue
            adapter = source if name == "native_source" else target
            cache = (translator if name == "translated" else closed_form).cache(features) if name in {
                "translated", "closed_form"} else None
            eos = adapter.model.generation_config.eos_token_id or tokenizer.eos_token_id
            ids, completed = greedy_ids(adapter, prompt.to(adapter.device), args.max_new_tokens, eos, cache)
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            observation["answers"][name] = {"text": answer, "eos_reached": completed, "generated_tokens": len(ids),
                                           **grade_response(official, case, answer, args.seed)}
            del cache
            report["task_quality"] = summarize_tasks(report["observations"], args.max_accuracy_loss, args.seed, conditions=conditions)
            save_report(args.output, report)
            print(f"{case.key} {name}: tokens={len(ids)} strict={observation['answers'][name]['prompt_passed']}", flush=True)
        del features, prompt
    report["task_quality"] = summarize_tasks(report["observations"], args.max_accuracy_loss, args.seed, conditions=conditions)
    report["status"] = "complete"
    save_report(args.output, report)
    print(f"Report: {args.output}; scores={report['task_quality']['scores']}", flush=True)
    return 1 if report["task_quality"]["observed_budget_met"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
