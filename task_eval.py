"""Generate-and-parse accuracy on ARC-Challenge and GSM8K, handing off at the prompt or after the reasoning trace."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time
from urllib.request import urlopen

import torch
import transformers

from ifeval import greedy_ids, open_report, paired_interval, save_report
from model_pair import implementation_fingerprints, load_pair, pair_args
from qwen_adapter import digest
from runtime import sha256_file
from translator import load_translator

ARC_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc"
            "&config=ARC-Challenge&split=test&offset={}&length=100")
GSM8K_FILE = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl")
TASKS = {
    "arc": {"sha256": "86ccaace4cd159b5c02b6b3339ebe8a7022a54f1b488e433c060b3bc38ac1f63", "items": 1172,
            "limit": None, "reasoning": "off", "budget": {"off": 1024, "on": 8192}},
    "gsm8k": {"sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14", "items": 1319,
              "limit": 256, "reasoning": "on", "budget": {"off": 2048, "on": 2048}},
}
NATIVE_CONDITIONS = ("native_source", "native_target")
TRACE_CONDITIONS = ("trace_text", "closed_form_trace", "translated_trace")
NUMBER = re.compile(r"(?<!\d)-?\d+(?:\.\d+)?")


def fetch_task(name):
    if name == "gsm8k":
        with urlopen(GSM8K_FILE, timeout=60) as response:
            return response.read()
    rows, total = [], 1
    while len(rows) < total:
        with urlopen(ARC_ROWS.format(len(rows)), timeout=60) as response:
            page = json.load(response)
        if not page["rows"]:
            raise ValueError("ARC-Challenge rows ended before the reported total")
        rows += [item["row"] for item in page["rows"]]
        total = page["num_rows_total"]
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()


def prepare_task(name, directory):
    data = fetch_task(name)
    if hashlib.sha256(data).hexdigest() != TASKS[name]["sha256"]:
        raise ValueError(f"{name}: upstream contents differ from the pinned checksum")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_bytes(data)
    print(f"Prepared {name} at {directory}", flush=True)


def arc_case(row):
    # Some rows label options 1-4; position gives one letter scheme for all of them.
    choices = row["choices"]["text"]
    letters = "ABCDE"[:len(choices)]
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))
    return {"key": row["id"], "choices": choices, "gold": letters[row["choices"]["label"].index(row["answerKey"])],
            "prompt": f"{row['question']}\n\n{options}\n\nAnswer with the letter of the correct option."}


def gsm8k_case(index, row):
    return {"key": f"gsm8k-{index}", "choices": None, "gold": parse_number(row["answer"].rsplit("####", 1)[1]),
            "prompt": f"{row['question']}\n\nGive the final answer as a number at the end, after '####'."}


def load_task(name, directory, limit):
    path = directory / f"{name}.jsonl"
    if sha256_file(path) != TASKS[name]["sha256"]:
        raise ValueError(f"{name}: prepared data differs from the pinned checksum; run --prepare")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cases = [arc_case(row) for row in rows] if name == "arc" else [gsm8k_case(i, row) for i, row in enumerate(rows)]
    if len(cases) != TASKS[name]["items"] or len({case["key"] for case in cases}) != len(cases):
        raise ValueError(f"{name}: expected {TASKS[name]['items']} uniquely keyed items")
    # A stable prefix of a hash ordering; chosen before observing any model answers.
    cases.sort(key=lambda case: digest(case["key"]))
    limit = TASKS[name]["limit"] if limit is None else limit
    if limit is not None and not 1 <= limit <= len(cases):
        raise ValueError(f"Limit must be between 1 and {len(cases)}")
    return cases[:limit]


def parse_choice(text, choices):
    letters = "ABCDE"[:len(choices)]
    # shortcut: a capital A before a lowercase word reads as the article unless "is" or "because" follows; widen from observed replies.
    letter = rf"(?<![\w'])([{letters}])(?![\w'])(?!(?<=A)\s+(?!is\b|because\b)[a-z])"
    # An explanation may discuss other options, so a stated answer outranks any option mention.
    marked = (re.findall(rf"(?:(?i:answer)(?:\s+is)?\W{{0,8}}|\\boxed\{{(?:\\text\{{)?){letter}", text) or
              re.findall(rf"{letter}\W{{0,4}}\s+is\s+the\s+(?i:correct|best|right)\b", text))
    if marked:
        return marked[-1]
    # A reply may commit in bold and then review every option.
    bolded = re.findall(rf"\*\*{letter}", text)
    if bolded:
        return bolded[0]
    marked = re.findall(rf"(?i:option|choice)(?:\s+is)?\W{{0,8}}{letter}", text)
    if marked:
        return marked[-1]
    mentioned = set(re.findall(letter, text))
    if len(mentioned) == 1:
        return mentioned.pop()
    named = [letter for letter, choice in zip(letters, choices) if choice.lower() in text.lower()]
    return named[0] if len(named) == 1 else None


def parse_number(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # A marker counts only directly before a number: "#### Step 3" is a Markdown heading.
    found = re.findall(rf"(?:####|\\boxed\{{|(?i:answer is))[\s:*$\\]*({NUMBER.pattern})", text)[-1:] or NUMBER.findall(text)[-1:]
    return format(Decimal(found[0]).normalize(), "f") if found else None


def answer_ids(ids, think_end, prefix_closed):
    """Tokens after the last closing marker; an open reasoning block has no answer yet."""
    if think_end in ids:
        return ids[len(ids) - ids[::-1].index(think_end):]
    return ids if prefix_closed else []


def trace_ids(ids, think_end, separator):
    """Source reasoning up to its closing marker, closed by force when the budget ended it first."""
    closed = think_end in ids
    return (ids[:ids.index(think_end) + 1] if closed else [*ids, think_end]) + separator, closed


def summarize(observations, conditions, seed=42):
    complete = [o for o in observations if set(o["answers"]) == set(conditions)]
    if not complete:
        return None
    scores = {}
    for name in conditions:
        answers = [o["answers"][name] for o in complete]
        accuracy = sum(a["correct"] for a in answers) / len(answers)
        scores[name] = {"accuracy": accuracy, "correct": sum(a["correct"] for a in answers),
                        "standard_error": math.sqrt(accuracy * (1 - accuracy) / len(answers)),
                        "unparsed": sum(a["parsed"] is None for a in answers),
                        # A map that favours one letter can look accurate on a subset rich in that letter.
                        "letter_counts": dict(Counter(a["parsed"] for a in answers if a["parsed"] and a["parsed"].isalpha())),
                        "token_cap_reached": sum(not a["eos_reached"] for a in answers),
                        "generated_tokens": sum(a["generated_tokens"] for a in answers),
                        "generation_seconds": sum(a["generation_seconds"] for a in answers)}
    candidate = next(name for name in ("translated", "translated_trace", "trace_text", "native_target")
                     if name in conditions)
    comparisons = {}
    for baseline in (name for name in conditions if name != candidate):
        differences = [int(o["answers"][candidate]["correct"]) - int(o["answers"][baseline]["correct"]) for o in complete]
        comparisons[baseline] = {"accuracy_delta": sum(differences) / len(differences),
                                 "paired_95_percent_interval": paired_interval(differences, seed)}
    return {"items_scored": len(complete), "scores": scores, candidate + "_vs": comparisons,
            "gold_letter_counts": dict(Counter(o["gold"] for o in complete if o["gold"].isalpha())),
            "traces_not_closed": sum(o.get("trace_closed") is False for o in complete)}


def load_maps(args, source, target):
    translator, metadata = load_translator(args.translator, source, target)
    baseline = args.closed_form or args.translator.with_name(args.translator.stem + "-closed-form" + args.translator.suffix)
    closed_form, baseline_metadata = load_translator(baseline, source, target)
    if (baseline_metadata["stage"] != "closed_form" or
            baseline_metadata["training_groups"] != metadata["training_groups"] or
            baseline_metadata["data_manifest_sha256"] != metadata["data_manifest_sha256"]):
        raise ValueError("Closed-form baseline does not belong to this training run")
    return translator, closed_form, metadata, sha256_file(baseline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pair_args(parser)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--prepare", action="store_true", help="Download the pinned task data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tasks"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--translator", type=Path)
    mode.add_argument("--native-only", action="store_true", help="Score native paths without translator artifacts")
    parser.add_argument("--closed-form", type=Path)
    parser.add_argument("--handoff", choices=["prompt", "trace"], default="prompt",
                        help="trace: the source reasons and the target writes only the answer")
    parser.add_argument("--reasoning", choices=["off", "on"], help="Default: arc off, gsm8k on; trace requires on")
    parser.add_argument("--limit", type=int, help="Fixed hash-selected subset; default arc all 1172, gsm8k 256")
    parser.add_argument("--max-new-tokens", type=int, help="Default: arc 1024 off, 8192 on; gsm8k 2048")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/task-eval.json"))
    args = parser.parse_args(argv)
    if args.prepare:
        prepare_task(args.task, args.data_dir)
        return 0
    settings = TASKS[args.task]
    args.reasoning = args.reasoning or ("on" if args.handoff == "trace" else settings["reasoning"])
    args.max_new_tokens = args.max_new_tokens or settings["budget"][args.reasoning]
    if not (args.translator or args.native_only) or args.max_new_tokens < 1:
        parser.error("Use --translator or --native-only, and a positive token budget")
    if args.native_only and args.closed_form is not None:
        parser.error("--closed-form applies only to translator evaluation")
    if args.handoff == "trace" and args.reasoning == "off":
        parser.error("Trace hand-off needs --reasoning on")
    cases = load_task(args.task, args.data_dir, args.limit)
    if args.handoff == "trace":
        conditions = (*NATIVE_CONDITIONS, *(TRACE_CONDITIONS[:1] if args.native_only else TRACE_CONDITIONS))
    else:
        conditions = NATIVE_CONDITIONS if args.native_only else (*NATIVE_CONDITIONS, "closed_form", "translated")
    source, target, tokenizer, provenance = load_pair(args)
    identity = {"case_keys": [case["key"] for case in cases], "dataset_sha256": settings["sha256"],
                "models": provenance, "devices": {"source": str(source.device), "target": str(target.device)},
                "implementation_sha256": implementation_fingerprints() | {Path(__file__).name: sha256_file(Path(__file__))},
                "settings": {k: v for k, v in vars(args).items() if k not in
                             {"prepare", "resume", "output", "data_dir", "translator", "closed_form"}},
                "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                "torch": str(torch.__version__), "transformers": transformers.__version__}}
    used, maps = set(), {}
    if not args.native_only:
        translator, closed_form, metadata, baseline_sha256 = load_maps(args, source, target)
        maps = {"closed_form": closed_form, "translated": translator,
                "closed_form_trace": closed_form, "translated_trace": translator}
        identity |= {"translator_sha256": sha256_file(args.translator), "closed_form_sha256": baseline_sha256}
        used = set(metadata["training_prompt_sha256"] + metadata["validation_prompt_sha256"])
    report = open_report(args.output, identity, args.resume)
    report.pop("paper_protocol", None)
    report["benchmark"] = f"{args.task}_reasoning_{args.reasoning}_{args.handoff}_handoff"
    report["protocol_deviations"] = (["prompt wording and answer parser are local choices"] +
                                     (["fixed subset"] if args.limit is not None else []) +
                                     ([] if args.native_only else ["one trained artifact"]))
    if report["status"] == "complete":
        print(f"Already complete: {args.output}", flush=True)
        return 0
    if not args.native_only:
        report["training"] = metadata
    save_report(args.output, report)
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    grade = (lambda case, text: parse_choice(text, case["choices"])) if args.task == "arc" else (
        lambda case, text: parse_number(text))
    observations = {o["key"]: o for o in report["observations"]}
    for case in cases:
        observation = observations.get(case["key"])
        if observation is not None and set(observation["answers"]) == set(conditions):
            continue
        text = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=args.reasoning == "on")
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if digest(prompt_ids) in used:
            raise ValueError(f"{case['key']}: prompt overlaps training/validation")
        if observation is None:
            observation = {"key": case["key"], "prompt": case["prompt"], "gold": case["gold"],
                           "prompt_tokens": len(prompt_ids), "answers": {}}
            report["observations"].append(observation)
        features = None
        for name in conditions:
            if name in observation["answers"]:
                continue
            adapter = source if name == "native_source" else target
            in_trace = name in TRACE_CONDITIONS
            prefix = prompt_ids + observation["trace_ids"] if in_trace else prompt_ids
            cache = None
            if name in maps:
                # The prefix is the same for every map on one item, so one source pass serves them all.
                if features is None:
                    features = source.capture(torch.tensor([prefix[:-1]], device=source.device))
                cache = maps[name].cache(features)
            eos = adapter.model.generation_config.eos_token_id or tokenizer.eos_token_id
            started = time.perf_counter()
            ids, completed = greedy_ids(adapter, torch.tensor([prefix], device=adapter.device),
                                        args.max_new_tokens, eos, cache)
            seconds = time.perf_counter() - started
            answer = tokenizer.decode(answer_ids(ids, think_end, in_trace or args.reasoning == "off"),
                                      skip_special_tokens=True)
            parsed = grade(case, answer)
            observation["answers"][name] = {"text": tokenizer.decode(ids, skip_special_tokens=True), "parsed": parsed,
                                            "correct": parsed == case["gold"], "eos_reached": completed,
                                            "generated_tokens": len(ids), "generation_seconds": seconds}
            if name == "native_source" and args.handoff == "trace":
                observation["trace_ids"], observation["trace_closed"] = trace_ids(ids, think_end, separator)
            del cache
            report["task_quality"] = summarize(report["observations"], conditions, args.seed)
            save_report(args.output, report)
            print(f"{case['key']} {name}: tokens={len(ids)} parsed={parsed} correct={parsed == case['gold']}", flush=True)
        del features
    report["task_quality"] = summarize(report["observations"], conditions, args.seed)
    report["status"] = "complete"
    save_report(args.output, report)
    print(f"Report: {args.output}; scores={report['task_quality']['scores']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
