"""Run the current Qwen3 head-wise experiment: validation-only LR selection, then three seeds.

Consumes prepared Nemotron/IFEval data. Each subprocess owns its model allocation;
the runner records commands and artifact hashes so interrupted work can be resumed.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

from model_pair import implementation_fingerprints, pair_args
from runtime import choose_device, sha256_file


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
CONDITIONS = ("native_source", "native_target", "closed_form", "translated")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def item_statistics(values):
    return {"mean": statistics.mean(values),
            "item_standard_error": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def aggregate_ifeval(reports):
    """Appendix E.1: average seeds per item before computing the item-level SE."""  
    if len(reports) != 3 or [r["identity"]["settings"]["seed"] for r in reports] != list(SEEDS):
        raise ValueError("Expected complete IFEval reports in seed order 42, 43, 44")
    reference = reports[0]
    keys = reference["identity"]["case_keys"]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("IFEval case keys must be nonempty and unique")
    rows = []
    for report in reports:
        if report["status"] != "complete" or report["benchmark"] != "ifeval_strict_cache_handoff":
            raise ValueError("Cannot aggregate incomplete or native-only evaluation")
        for field in ("case_keys", "dataset_manifest", "implementation_sha256", "environment"):
            if report["identity"][field] != reference["identity"][field]:
                raise ValueError(f"IFEval {field} differs across seeds")
        settings = {k: v for k, v in report["identity"]["settings"].items() if k != "seed"}
        expected = {k: v for k, v in reference["identity"]["settings"].items() if k != "seed"}
        if settings != expected or report["models"] != reference["models"]:
            raise ValueError("IFEval models or generation settings differ across seeds")
        items = {o["key"]: o for o in report["observations"]}
        if len(items) != len(report["observations"]) or set(items) != set(keys):
            raise ValueError("IFEval items are missing or duplicated")
        if any(set(o["answers"]) != set(CONDITIONS) for o in items.values()):
            raise ValueError("IFEval conditions are incomplete")
        rows.append(items)
    scores, per_item = {}, {}
    for condition in CONDITIONS:
        values, instruction_values, capped = [], [], []
        for key in keys:
            cases = [row[key] for row in rows]
            for field in ("prompt", "prompt_tokens", "instruction_id_list", "kwargs"):
                if any(case[field] != cases[0][field] for case in cases):
                    raise ValueError("IFEval item contents differ across seeds")
            answers = [case["answers"][condition] for case in cases]
            count = len(cases[0]["instruction_id_list"])
            if not count or any(len(a["instructions_passed"]) != count for a in answers):
                raise ValueError("IFEval instruction outcomes do not match the item")
            values.append(statistics.mean(int(a["prompt_passed"]) for a in answers))
            instruction_values.extend(statistics.mean(int(a["instructions_passed"][i]) for a in answers)
                                      for i in range(count))
            capped.append(sum(not a["eos_reached"] for a in answers))
        per_item[condition] = values
        scores[condition] = {"strict_prompt_accuracy": item_statistics(values),
                             "strict_instruction_accuracy": statistics.mean(instruction_values),
                             "token_cap_hits_across_seeds": sum(capped),
                             "per_seed_prompt_accuracy": {str(seed): report["task_quality"]["scores"][condition]["prompt_accuracy"]
                                                          for seed, report in zip(SEEDS, reports)}}
    return {"prompts": len(keys), "seeds": list(SEEDS), "scores": scores,
            "translated_minus_native_target": item_statistics(
                [a - b for a, b in zip(per_item["translated"], per_item["native_target"])]),
            "per_item_seed_mean": [{"key": key, **{c: per_item[c][i] for c in CONDITIONS}} for i, key in enumerate(keys)],
            "uncertainty": "sample SE over items after seed averaging; not training-seed uncertainty",
            "quality_acceptance_budget": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pair_args(parser)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ifeval-data-dir", type=Path, default=Path("data/ifeval"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-samples", type=int, default=400)
    parser.add_argument("--distill-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--continuation-tokens", type=int, default=2048)
    parser.add_argument("--lr-center", type=float, default=1e-3)
    parser.add_argument("--max-edge-extensions", type=int, default=4,
                        help="Stop unresolved after this many extra rates; never select an edge winner")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--ifeval-limit", type=int, default=541)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-examples", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42 or (args.source_model, args.target_model) != ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        parser.error("This recipe covers pretrained Qwen3-0.6B -> 1.7B, selecting rates with seed 42")
    if any(not re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (args.source_revision, args.target_revision)):
        parser.error("Use immutable source/target revisions (the Makefile supplies them)")
    if (min(args.fit_samples, args.distill_steps, args.batch_size, args.continuation_tokens,
            args.checkpoint_every, args.repeats, args.timing_examples, args.cpu_threads) < 1
            or args.warmup < 0 or args.max_edge_extensions < 0 or not 1 <= args.ifeval_limit <= 541
            or not math.isfinite(args.lr_center) or args.lr_center <= 0):
        parser.error("Invalid experiment count, budget or learning rate")
    for name in ("data_dir", "ifeval_data_dir", "artifact_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    data = json.loads((args.data_dir / "manifest.json").read_text())
    if data["counts"]["validation"] != 64 or data["counts"]["train"] < args.fit_samples:
        parser.error("Rate selection requires 64 validation conversations and enough fitting samples")
    for split in ("train", "validation", "test"):
        filename = split + ".jsonl"
        if sha256_file(args.data_dir / filename) != data["files"][filename]:
            raise ValueError(f"Prepared {split} checksum differs from its manifest")
    identity = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"},
                "implementation_sha256": implementation_fingerprints(),
                "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
                "ifeval_manifest_sha256": sha256_file(args.ifeval_data_dir / "manifest.json"),
                "python": platform.python_version(), "platform": platform.platform(),
                "resolved_device": str(choose_device(args.device))}
    state_path = args.output_dir / "recipe.json"
    if state_path.exists():
        if not args.resume:
            parser.error("Recipe already exists; use --resume or fresh artifact/output directories")
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("Recipe resume identity differs")
    else:
        if args.resume:
            parser.error("No recipe report to resume")
        state = {"schema_version": 1, "identity": identity, "status": "incomplete", "jobs": {},
                 "selection": {}, "full_paper_reproduction": False, "performance_claim_validated": False,
                 "scope": "one direction, head-wise Qwen3-0.6B -> 1.7B; Nemotron LM and IFEval",
                 "data_manifest": data}
        write_json(state_path, state)

    def job(name, command, outputs, resume_flag=None, allowed=(0,)):
        command = [str(v) for v in command]
        saved = state["jobs"].get(name)
        if saved is not None:
            if saved["command"] != command or saved["outputs"] != {str(p): sha256_file(p) for p in outputs}:
                raise ValueError(f"Completed job changed: {name}")
            return saved["exit_code"]
        actual = command + (["--resume"] if resume_flag is not None and resume_flag.exists() else [])
        print(f"Recipe: {name}", flush=True)
        result = subprocess.run(actual, cwd=ROOT)
        if result.returncode not in allowed:
            raise RuntimeError(f"{name} failed with exit {result.returncode}; partial artifacts retained")
        state["jobs"][name] = {"command": command, "exit_code": result.returncode,
                               "outputs": {str(p): sha256_file(p) for p in outputs}}
        write_json(state_path, state)
        return result.returncode

    def pair(seed):
        command = []
        for name in ("source_model", "target_model", "source_revision", "target_revision", "device", "dtype", "attention", "cpu_threads"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        return command + ["--seed", str(seed)] + (["--local-files-only"] if args.local_files_only else [])

    def train(seed, rate, initialization=None):
        name = f"seed{seed}-lr{rate:.12g}"
        artifact = args.artifact_dir / name / "translator.pt"
        baseline = artifact.with_name("translator-closed-form.pt")
        metadata = artifact.with_suffix(".training.json")
        command = [sys.executable, ROOT / "train_translator.py", *pair(seed), "--data-dir", args.data_dir,
                   "--fit-samples", args.fit_samples, "--distill-steps", args.distill_steps, "--batch-size", args.batch_size,
                   "--continuation-tokens", args.continuation_tokens, "--learning-rate", repr(rate),
                   "--checkpoint-every", args.checkpoint_every, "--output", artifact]
        if initialization is not None:
            command += ["--init-from", initialization]
        job("train-" + name, command, [artifact, baseline, metadata], artifact.with_suffix(".checkpoint.pt"))
        record = json.loads(metadata.read_text())
        if record["completed_distill_steps"] != args.distill_steps:
            raise ValueError("Incomplete training cannot enter rate selection")
        score = record["final_validation"]["mean_kl_nats"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite final validation KL")
        return artifact, baseline, score

    native = args.output_dir / "ifeval-native.json"
    job("native-ifeval", [sys.executable, ROOT / "ifeval.py", "ifeval", "--native-only", *pair(42),
                         "--data-dir", args.ifeval_data_dir, "--limit", args.ifeval_limit, "--output", native], [native], native)
    # Stage 1 is deterministic for the fixed fitting set; reuse its maps in every cell.
    _, initialization, _ = train(42, args.lr_center)
    offsets = set(range(-2, 3))
    trials = {}
    while True:
        for offset in sorted(offsets - trials.keys()):
            rate = args.lr_center * 10 ** (offset / 2)
            artifact, _, kl = train(42, rate, initialization if offset else None)
            trials[offset] = {"learning_rate": rate, "validation_kl_nats": kl, "artifact": str(artifact)}
        # Prefer an interior rate for exact ties; no test/IFEval score enters selection.
        best = min(trials, key=lambda k: (trials[k]["validation_kl_nats"], k in (min(offsets), max(offsets)), abs(k), k))
        interior = min(offsets) < best < max(offsets)
        state["selection"] = {"criterion": "lowest final validation KL on 64 held-out conversations",
                              "seed": 42, "trials": [trials[k] for k in sorted(trials)], "interior_winner": interior}
        write_json(state_path, state)
        if interior:
            break
        if len(offsets) - 5 >= args.max_edge_extensions:
            state["status"] = "selection_unresolved"
            write_json(state_path, state)
            raise RuntimeError("Best rate remains on an edge; no rate selected. Widen the declared search in a new experiment.")
        offsets.add(min(offsets) - 1 if best == min(offsets) else max(offsets) + 1)
    selected = trials[best]["learning_rate"]
    state["selection"]["selected_learning_rate"] = selected
    write_json(state_path, state)
    reports, measurements = [], []
    for seed in SEEDS:
        artifact, _, _ = train(seed, selected, initialization if seed != 42 or best != 0 else None)
        output = args.output_dir / f"seed{seed}"
        lm, task = output / "nemotron.json", output / "ifeval.json"
        # A diagnostic failure returns 1 *after* writing evidence. Fatal errors must not
        # accidentally reuse a stale report, so remove only this unfinished job's output.
        if f"benchmark-{seed}" not in state["jobs"] and lm.exists():
            lm.unlink()
        job(f"benchmark-{seed}", [sys.executable, ROOT / "benchmark.py", "nemotron", *pair(seed),
            "--translator", artifact, "--data-dir", args.data_dir, "--continuation-tokens", args.continuation_tokens,
            "--repeats", args.repeats, "--warmup", args.warmup, "--timing-examples", args.timing_examples,
            "--output", lm], [lm], allowed=(0, 1))
        job(f"ifeval-{seed}", [sys.executable, ROOT / "ifeval.py", "ifeval", *pair(seed), "--translator", artifact,
            "--data-dir", args.ifeval_data_dir, "--limit", args.ifeval_limit, "--output", task], [task], task)
        reports.append(json.loads(task.read_text()))
        measurements.append(json.loads(lm.read_text()))
    state["ifeval"] = aggregate_ifeval(reports)
    state["nemotron_per_seed"] = {str(seed): {"quality": report["quality"], "verification": report["verification"],
        "report": str(args.output_dir / f"seed{seed}" / "nemotron.json")} for seed, report in zip(SEEDS, measurements)}
    state["status"] = "complete"
    state["diagnostics_passed"] = all(not r["verification"]["failed_checks"] for r in measurements)
    write_json(state_path, state)
    print(f"Recipe report: {state_path}; completed, diagnostics_passed={state['diagnostics_passed']}", flush=True)
    return 0 if state["diagnostics_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
