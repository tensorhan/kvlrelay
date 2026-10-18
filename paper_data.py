"""Prepare a traceable subset of KV-Lingo's Nemotron chat stream.

Paper §4.1, Table 4, Appendix C: natural and packed assistant-turn samples.
Length ranges, reasoning balance, component identities and deviations are recorded.
"""

import argparse
from contextlib import closing
from http.client import IncompleteRead
import json
from pathlib import Path
import random
import re
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_url
from transformers import AutoTokenizer

from runtime import sha256_file
from model_pair import tokenizer_fingerprint
from qwen_adapter import digest


DATASET = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2"
PAPER = "https://arxiv.org/html/2609.32610v1"


def stream_records(url, filename):
    """Resume interrupted transfers at the last complete JSONL record boundary."""
    offset = line_number = failures = 0
    total = etag = None
    while True:
        request = Request(url, headers={"Range": f"bytes={offset}-"}) if offset else url
        try:
            with closing(urlopen(request, timeout=90)) as response:
                headers = getattr(response, "headers", {})
                if offset:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
                    if getattr(response, "status", None) != 206 or not match or int(match[1]) != offset:
                        raise ValueError(f"{filename}: server did not honor the resume offset")
                    if total is not None and int(match[3]) != total:
                        raise ValueError(f"{filename}: dataset size changed during transfer")
                    total = int(match[3])
                elif headers.get("Content-Length"):
                    total = int(headers["Content-Length"])
                current_etag = headers.get("ETag")
                if etag is not None and current_etag != etag:
                    raise ValueError(f"{filename}: dataset identity changed during transfer")
                etag = current_etag
                for line in response:
                    if not line.endswith(b"\n") and total is not None and offset + len(line) < total:
                        raise IncompleteRead(line, total - offset - len(line))
                    line_number += 1
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise ValueError(f"{filename}:{line_number}: invalid complete JSONL record") from exc
                        offset += len(line)
                        failures = 0
                        yield record, filename, line_number
                    else:
                        offset += len(line)
                if total is not None and offset != total:
                    raise IncompleteRead(b"", total - offset)
                return
        except (OSError, IncompleteRead) as exc:
            failures += 1
            if failures > 3:
                raise RuntimeError(f"{filename}: transfer failed repeatedly at byte {offset}") from exc
            print(f"Resuming {filename} at byte {offset} after interrupted transfer", flush=True)


def clean_content(message):
    return re.sub(r"<think>.*?</think>\s*", "", message.get("content", ""), flags=re.DOTALL)


def conversation_group(record):
    # Conservative root grouping keeps regenerated answers and conversation
    # extensions together even when their uuid or assistant text changes.
    root = []
    for message in record["messages"]:
        if message["role"] in ("system", "user"):
            root.append({"role": message["role"], "content": clean_content(message)})
        if message["role"] == "user":
            break
    return digest(root)


def partition(group):
    bucket = int(group[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def render_record(record, tokenizer, seed, *, history=None):
    messages = record["messages"]
    candidates = [i for i, m in enumerate(messages) if m["role"] == "assistant" and i > 0
                  and messages[i - 1]["role"] == "user"]
    if not candidates:
        return None
    group = conversation_group(record)
    index = random.Random(f"{seed}:{group}").choice(candidates)
    reply = messages[index]
    block = re.search(r"<think>(.*?)</think>", reply.get("content", ""), flags=re.DOTALL)
    reasoning = bool((reply.get("reasoning_content") or "").strip()) or bool(block and block[1].strip())
    if history is None:
        history = [{"role": m["role"], "content": clean_content(m)} for m in messages[:index]]
    prefix = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=reasoning)
    answer = reply.get("content", "")
    if not reasoning:
        answer = clean_content(reply)
    if reply.get("reasoning_content") and "<think>" not in answer:
        answer = "<think>\n" + reply["reasoning_content"] + "\n</think>\n\n" + answer
    if not answer:
        return None
    continuation = answer + "<|im_end|>"
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    combined = tokenizer.encode(prefix + continuation, add_special_tokens=False)
    if combined[:len(prompt_ids)] != prompt_ids:
        # Do not silently change the boundary tokens or score a template placeholder.
        return None
    return {"id": record.get("uuid", group), "group_sha256": group,
            "assistant_turn": index, "reasoning": "on" if reasoning else "off",
            "prompt_ids": prompt_ids, "continuation_ids": combined[len(prompt_ids):],
            "rendered_text_sha256": digest([prefix, continuation])}



def sample_groups(example):
    """Include every contributing conversation, not just a packed sample's new ID."""
    components = example.get("components", [])
    if example.get("kind") == "packed" and not components:
        raise ValueError("Packed samples require component provenance")
    return {example["group_sha256"]} | {c["group_sha256"] for c in components}


def pack_samples(pool, tokenizer, seed, mode, minimum, maximum):
    """Move one selected question/reply out of whole, split-local conversations."""
    rng = random.Random(seed)
    eligible = [c for c in pool if c["sample"]["reasoning"] == mode]
    if not eligible:
        return None
    target = rng.choice(eligible)
    target_group = target["sample"]["group_sha256"]
    others = [c for c in pool if c["sample"]["group_sha256"] != target_group]
    rng.shuffle(others)
    selected, used = [target], {target_group}
    desired = rng.randint(minimum, maximum)

    def render(components):
        history = []
        for component in components:
            turn = component["sample"]["assistant_turn"]
            messages = component["record"]["messages"]
            # Earlier answers stay in context; the selected question and its own
            # answer are absent until the question is reinserted at the end.
            end = turn - 1 if component is target else turn + 1
            history.extend({"role": m["role"], "content": clean_content(m)} for m in messages[:end])
        question = target["record"]["messages"][target["sample"]["assistant_turn"] - 1]
        history.append({"role": "user", "content": clean_content(question)})
        return render_record(target["record"], tokenizer, target["seed"], history=history)

    result = render(selected)
    if result is None or len(result["prompt_ids"]) > maximum:
        return None
    for candidate in others:
        group = candidate["sample"]["group_sha256"]
        if group in used:
            continue
        proposed = render(selected + [candidate])
        if proposed is None or len(proposed["prompt_ids"]) > maximum:
            continue
        selected.append(candidate)
        used.add(group)
        result = proposed
        if len(result["prompt_ids"]) >= desired:
            break
    if len(selected) < 2 or len(result["prompt_ids"]) < minimum:
        return None
    provenance = [{k: c["sample"][k] for k in
                   ("id", "group_sha256", "assistant_turn", "dataset_file", "dataset_row")}
                  for c in selected]
    signature = digest({"components": provenance, "target": target_group, "reasoning": mode})
    return result | {"id": "packed-" + signature, "group_sha256": signature, "kind": "packed",
                     "components": provenance, "target_component_group": target_group,
                     "desired_prefix_tokens": desired}


def collect_samples(streams, tokenizer, wanted, args):
    modes = tuple(streams)
    kinds = ("natural", "packed") if args.mixture == "mixed" else ("natural",)
    strata = [(kind, mode) for kind in kinds for mode in modes]
    quotas = {split: {s: n // len(strata) + (i < n % len(strata))
                     for i, s in enumerate(strata)} for split, n in wanted.items()}
    samples = {split: [] for split in wanted}
    counts = {split: {s: 0 for s in strata} for split in wanted}
    pools = {split: [] for split in wanted}
    seen = set()
    prompts = set()
    natural_max = min(args.natural_max_tokens, args.max_prefix_tokens)
    # Equal quotas in linear token-length bins approximate the natural half's
    # uniform distribution without truncating conversations or answers.
    bins = {split: {mode: min(8, quotas[split][("natural", mode)]) for mode in modes}
            for split in wanted}
    bin_counts = {}
    scanned = skipped = 0

    def add(sample, split, kind):
        signature = digest(sample["prompt_ids"])
        if signature in prompts:
            return False
        sample["continuation_ids"] = sample["continuation_ids"][:args.continuation_tokens]
        sample["kind"] = kind
        samples[split].append(sample)
        counts[split][(kind, sample["reasoning"])] += 1
        prompts.add(signature)
        print(f"{split}: {len(samples[split])}/{wanted[split]} ({kind}, "
              f"{len(sample['prompt_ids'])} tokens, {sample['reasoning']})", flush=True)
        return True

    while any(len(samples[s]) < wanted[s] for s in wanted) and scanned < args.max_scan:
        mode = modes[scanned % len(modes)]
        try:
            record, filename, row = next(streams[mode])
        except StopIteration as exc:
            raise ValueError("Dataset exhausted before filling the declared mixture/length quotas") from exc
        scanned += 1
        sample = render_record(record, tokenizer, args.seed)
        if sample is None or sample["reasoning"] != mode:
            skipped += 1
            continue
        root = sample["group_sha256"]
        split = partition(root)  # Assign BEFORE any packing.
        if (root, mode) in seen or len(samples[split]) >= wanted[split]:
            skipped += 1
            continue
        seen.add((root, mode))
        sample["dataset_file"], sample["dataset_row"] = filename, row
        length = len(sample["prompt_ids"])
        if counts[split][("natural", mode)] < quotas[split][("natural", mode)] and 10 <= length <= natural_max:
            allowed = True
            if args.mixture == "mixed":
                n = bins[split][mode]
                bucket = min(n - 1, (length - 10) * n // (natural_max - 9))
                quota = quotas[split][("natural", mode)]
                capacity = quota // n + (bucket < quota % n)
                key = (split, mode, bucket)
                allowed = bin_counts.get(key, 0) < capacity
            if allowed and add(dict(sample), split, "natural") and args.mixture == "mixed":
                bin_counts[key] = bin_counts.get(key, 0) + 1
        if args.mixture == "mixed":
            pools[split].append({"record": record, "sample": sample, "seed": args.seed})
            # Bound raw-conversation memory; retain the current deterministic window.
            pools[split] = pools[split][-256:]
            for target_mode in modes:
                if counts[split][("packed", target_mode)] >= quotas[split][("packed", target_mode)]:
                    continue
                packed = pack_samples(pools[split], tokenizer, f"{args.seed}:{split}:{scanned}:{target_mode}",
                                      target_mode, args.packed_min_tokens, args.max_prefix_tokens)
                if packed is not None:
                    add(packed, split, "packed")
    if any(len(samples[s]) != wanted[s] for s in wanted):
        raise ValueError("Unfilled mixture/length quotas; increase --max-scan or choose feasible length ranges")
    for split in samples:
        rows = {stratum: [e for e in samples[split] if (e["kind"], e["reasoning"]) == stratum]
                for stratum in strata}
        samples[split] = [rows[s][i] for i in range(max(map(len, rows.values())))
                          for s in strata if i < len(rows[s])]
    check_disjoint(*samples.values())
    return samples, scanned, skipped


def prepare(args):
    api = HfApi()
    revision = api.dataset_info(DATASET, revision=args.dataset_revision).sha
    model_revision = api.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=model_revision)
    files = api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
    modes = ("off", "on") if args.reasoning == "both" else (args.reasoning,)
    def records(mode):
        selected = [f for f in sorted(files) if "reasoning_" + mode in f and f.endswith(".jsonl")]
        if not selected:
            raise ValueError(f"No reasoning_{mode} JSONL files found at pinned dataset revision")
        for filename in selected:
            url = hf_hub_url(DATASET, filename, repo_type="dataset", revision=revision)
            yield from stream_records(url, filename)

    streams = {mode: iter(records(mode)) for mode in modes}
    wanted = {"train": args.train_samples, "validation": args.validation_samples, "test": args.test_samples}
    try:
        samples, scanned, skipped = collect_samples(streams, tokenizer, wanted, args)
    finally:
        for stream in streams.values():
            stream.close()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in samples.items():
        path = directory / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        hashes[path.name] = sha256_file(path)
    manifest = {
        "schema_version": 2, "paper": PAPER, "paper_sections": ["4.1", "Appendix C", "Table 4"],
        "dataset": DATASET, "dataset_revision": revision, "dataset_license": "ODC-By",
        "tokenizer": args.tokenizer, "tokenizer_revision": model_revision,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer), "seed": args.seed,
        "split_rule": "root hash: 10% test, 10% validation, 80% train; every packed component assigned before packing",
        "sample_selection": "deterministic assistant turns; reasoning quotas; mixed natural length-bin quotas; seeded whole-conversation packing",
        "mixture": args.mixture,
        "reasoning_requested": args.reasoning,
        "counts": {k: len(v) for k, v in samples.items()}, "files": hashes,
        "reasoning_counts": {k: {mode: sum(e["reasoning"] == mode for e in v) for mode in ("off", "on")} for k, v in samples.items()},
        "kind_counts": {k: {kind: sum(e["kind"] == kind for e in v) for kind in ("natural", "packed")} for k, v in samples.items()},
        "unique_component_conversations": {k: len({g for e in v for g in sample_groups(e)
                                                   if g != e["group_sha256"] or e["kind"] == "natural"})
                                           for k, v in samples.items()},
        "prefix_token_ranges": {k: [min(len(e["prompt_ids"]) for e in v), max(len(e["prompt_ids"]) for e in v)] for k, v in samples.items()},
        "scanned_records": scanned, "skipped_records": skipped,
        "max_prefix_tokens": args.max_prefix_tokens, "continuation_cap": args.continuation_tokens,
        "natural_prefix_range": [10, min(args.natural_max_tokens, args.max_prefix_tokens)],
        "packed_prefix_range": [args.packed_min_tokens, args.max_prefix_tokens] if args.mixture == "mixed" else None,
        "full_paper_reproduction": False,
        "deviations": (["natural-only subset; no packed half or controlled natural length distribution"]
                       if args.mixture == "natural" else [
                           "natural length-bin quotas approximate uniform sampling; packed contexts use whole-conversation greedy fits",
                           "bounded packing pool and deterministic local split; no exact authors' sample-ID reproduction"]) + (
                       ["shortened prefix/continuation ranges; reduced local training mixture"]
                       if args.max_prefix_tokens < 16000 or args.natural_max_tokens < 10000
                          or args.continuation_tokens < 2048 or args.reasoning != "both" else []),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared paper-data subset: {directory}", flush=True)


def load_examples(directory, split, tokenizer_hash, vocab_size):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") not in (1, 2) or manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Prepared data schema/tokenizer does not match this model pair")
    path = directory / (split + ".jsonl")
    if manifest["files"].get(path.name) != sha256_file(path):
        raise ValueError("Prepared data checksum mismatch")
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not examples or len(examples) != manifest["counts"][split]:
        raise ValueError("Prepared data sample count mismatch")
    for example in examples:
        if example.get("kind") == "packed":
            roots = [c["group_sha256"] for c in example.get("components", [])]
            if len(roots) < 2 or len(set(roots)) != len(roots) or any(partition(g) != split for g in roots):
                raise ValueError("Packed component provenance is missing, repeated or crosses data splits")
            if example.get("target_component_group") not in roots:
                raise ValueError("Packed target conversation is not a component")
        for key, minimum in (("prompt_ids", 2), ("continuation_ids", 1)):
            ids = example[key]
            if len(ids) < minimum or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
                raise ValueError("Invalid token IDs in prepared data")
    return examples, manifest


def check_disjoint(*splits):
    groups, prompts = set(), set()
    for examples in splits:
        current_groups = {group for e in examples for group in sample_groups(e)}
        current_prompts = {digest(e["prompt_ids"]) for e in examples}
        if groups & current_groups or prompts & current_prompts:
            raise ValueError("Conversation or prompt leakage across data splits")
        groups |= current_groups
        prompts |= current_prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["nemotron"])
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--reasoning", choices=["off", "on", "both"], default="both")
    parser.add_argument("--mixture", choices=["natural", "mixed"], default="natural")
    parser.add_argument("--natural-max-tokens", type=int, default=10000)
    parser.add_argument("--packed-min-tokens", type=int, default=12000)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--max-prefix-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/nemotron-pilot"))
    args = parser.parse_args(argv)
    if min(args.train_samples, args.validation_samples, args.test_samples, args.continuation_tokens, args.max_scan) < 1 or args.max_prefix_tokens < 10:
        parser.error("Positive sample/continuation/scan counts and prefix cap >= 10 are required")
    if args.natural_max_tokens < 10:
        parser.error("Natural prefix cap must be >= 10")
    if args.mixture == "mixed":
        strata = 4 if args.reasoning == "both" else 2
        if any(n % strata for n in (args.train_samples, args.validation_samples, args.test_samples)):
            parser.error("Mixed split counts must be divisible by four (both modes) or two (one mode)")
        if not 10 <= args.natural_max_tokens <= args.packed_min_tokens <= args.max_prefix_tokens:
            parser.error("Mixed ranges require 10 <= natural cap <= packed minimum <= overall prefix cap")
    prepare(args)


if __name__ == "__main__":
    main()
