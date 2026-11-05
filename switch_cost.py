"""Time the wait at a mid-chat model switch on contexts longer than the prepared chats."""

import argparse
import gc
import json
from pathlib import Path
import platform

import torch

from benchmark import timing_summary
from handoff_eval import example_tensors
from model_pair import load_pair, pair_args
from paper_data import load_examples
from runtime import timed
from translator import load_translator


def long_chat(examples, length):
    # Latency depends on length, not meaning, so held-out conversations are joined and cropped.
    ids = []
    for example in examples:
        ids += example["prompt_ids"] + example["continuation_ids"]
        if len(ids) > length:
            return {"id": f"joined-{length}", "prompt_ids": ids[:length], "continuation_ids": [ids[length]]}
    raise ValueError(f"Held-out data holds fewer than {length} tokens")


def measure_switch(translator, example, repeats, warmup=1):
    # Paths run apart, holding only what a real switch holds, so long contexts do not page on small machines.
    source, target = translator.source, translator.target
    prompt, _ = example_tensors(example, target, 1)
    def run(operation):
        times = []
        for index in range(warmup + repeats):
            gc.collect()
            if target.device.type == "mps":
                torch.mps.empty_cache()
            output, elapsed = timed(operation, target.device)
            if output.past_key_values.get_seq_length() != prompt.shape[1]:
                raise RuntimeError("Timed cache length mismatch")
            del output
            if index >= warmup:
                times.append(elapsed)
        return timing_summary(times)
    reread = run(lambda: target.forward(prompt))
    features = source.capture(prompt[:, :-1].to(source.device))
    switch = run(lambda: target.forward(prompt[:, -1:], start=prompt.shape[1] - 1, cache=translator.cache(features)))
    return {"prompt_tokens": prompt.shape[1], "timing_ms": {"target_reprefill": reread, "existing_source_cache_handoff": switch},
            "median_existing_cache_speedup": reread["median"] / switch["median"], "source_pre_norm_cache_bytes": features.nbytes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pair_args(parser)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--translator", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.lengths) < 2:
        parser.error("Context lengths must be >= 2")
    source, target, _, provenance = load_pair(args)
    translator, metadata = load_translator(args.translator, source, target)
    examples, _ = load_examples(args.data_dir, "test", provenance["target"]["tokenizer_sha256"], target.config.vocab_size)
    report = {"models": provenance, "translator_stage": metadata.get("stage"), "rows": [],
              "environment": {"platform": platform.platform(), "torch": str(torch.__version__),
                              "device": str(target.device), "dtype": args.dtype}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        row = measure_switch(translator, long_chat(examples, length), max(3, min(12, 40000 // length)))
        report["rows"].append(row)
        # Each length is saved as it finishes, so running out of memory keeps the shorter results.
        args.output.write_text(json.dumps(report, indent=1))
        print(f"{length}: re-read {row['timing_ms']['target_reprefill']['median']:.0f} ms, "
              f"switch {row['timing_ms']['existing_source_cache_handoff']['median']:.0f} ms, "
              f"{row['median_existing_cache_speedup']:.1f}x", flush=True)


if __name__ == "__main__":
    main()
