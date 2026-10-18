# KVLRelay

Experiment scripts for [KV-Lingo](https://arxiv.org/html/2609.32610v1): train a
directional KV-cache translator and compare a model handoff with native target
re-prefill. Implements head-wise linear maps for **Qwen3-0.6B → Qwen3-1.7B**.
An independent, partial reproduction; quality and speedup remain experimental.

## Run

Use Python 3.12.14, the version tested with the locked dependencies.

```sh
uv venv --python 3.12.14
uv sync --locked --extra ifeval
make run DEVICE=cuda                    # Pilot: prepare, train, evaluate
make prepare PROFILE=paper-budget
make recipe PROFILE=paper-budget DEVICE=cuda RUN=qwen-headwise
```

Set `DEVICE=mps` on Apple Silicon. The [Makefile](Makefile) pins the original
Hugging Face checkpoints, [Nemotron chat data](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2),
and the [official IFEval grader](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval).
`make prepare`, `train`, `benchmark` and `ifeval` also run independently.
`make ifeval-native DEVICE=cuda` scores all 541 prompts without a translator;
prepare its data with `.venv/bin/python ifeval.py ifeval --prepare` first.
Resume with `TRAIN_ARGS=--resume`, `IFEVAL_ARGS=--resume` or `RECIPE_ARGS=--resume`
on the corresponding target, keeping the same run name and settings.
Generated data, maps and reports live under `data/`, `artifacts/` and `results/`.

## Protocol

| Stage | Script | Output |
| --- | --- | --- |
| Prepare | `paper_data.py` | Disjoint natural/packed Nemotron splits; reasoning on/off; hashes |
| Train | `train_translator.py` | FP64 closed-form fit, then FP32 KL distillation; frozen models |
| Measure | `benchmark.py` | Held-out target KL, reply perplexity, native controls, four timing paths |
| Generate | `ifeval.py` | Official strict instruction accuracy; resumable answer-level report |
| Orchestrate | `recipe.py` | Validation-only LR selection, seeds 42/43/44, IFEval item-level mean/SE |

Both profiles use BF16 models by default (`DTYPE=float32` selects FP32).
The adapter supports batch-one, full-attention Qwen3 with matching layer/KV-head
counts, a shared tokenizer and default RoPE. Frozen model weights are unchanged;
only the maps are trained. Atomic checkpoints retain optimizer/RNG/update state
every 100 updates; resume rejects changed data, code, models or training settings.

| Profile | Train / validation / test | Fit; updates × batch | Prefix / reply cap | IFEval |
| --- | ---: | ---: | ---: | ---: |
| `pilot` | 256 / 64 / 64 | 256; 100 × 1 | 1,024 / 128 | 32 |
| `paper-budget` | 40,000 / 64 / 64 | 400; 5,000 × 8 | 16,384 / 2,048 | 541 |

`run` executes one training cell. `recipe` first scores native models, searches five
half-decade-spaced rates by final validation KL, extends edge winners, then trains
and evaluates three seeds. Closed-form initialization is reused across cells.
Local checks cover checkpoint recovery and recipe orchestration; the large recipe is unexecuted.
The 40,000-example corpus and initial LR window are our choices; sampling is independent.

KL measures closeness to the native target's token distribution; perplexity scores
the recorded reply. IFEval separately grades generated answers. Lower KL alone
does not establish useful transfer: compare task accuracy and latency against
both native models. Research runs have no default accuracy-loss gate; deployment
limits are optionally selectable with IFEval's `--max-accuracy-loss`.

Timing compares target re-prefill, existing-source handoff, source-prefill plus
handoff, and existing-target continuation through first-token logits. Both models
are resident; warm handoff requires existing source pre-normalization features.
Loading, tokenization and network transfer are excluded; reports retain raw trials.
Failed checks return nonzero after writing JSON; `make run` still collects IFEval.
CV ≤10% is our timing diagnostic screen.

## Recorded M3 pilot

**Apple M3, 16 GiB**: seed 42; 32 natural fitting conversations,
100 updates, eight held-out continuations (512/64-token caps); 32 IFEval prompts,
1,024 generated-token cap. This smaller run uses neither preset's full budget.

| Cache | Target KL BF16 ↓ | Target KL FP32 ↓ | IFEval BF16 /32 ↑ |
| --- | ---: | ---: | ---: |
| Native source | 0.5260 | 0.5227 | 14 |
| Native target | 0 | 0 | 13 |
| Closed-form | 0.8145 | 0.8194 | 9 |
| Distilled | 0.3666 | 0.3674 | 6 |

FP32 evaluation reused BF16-trained maps. Native controls passed 6/8 BF16 and 8/8
FP32; reconstruction exactly matched ordinary split execution in both precisions.
Timing stability passed 5/8 BF16 and 2/8 FP32 contexts (30 repeats, five warmups).
Distillation lowered KL by 55%, but task accuracy lost 21.9 points against the
target; 11/32 translated answers reached the output cap. No validated speedup.

## Where this can be taken forward

From what it's obvious: self-hosted model routing, coding-agent handoffs
and document QA over shared context. 

Since this requires access to `internal` KV tensors,
pair-specific quality/latency evidence and a native re-prefill fallback.

Ordinary hosted chat APIs cannot expose
this transfer path through a prompt alone.

Looking for contributors for larger runs, more pairs, corrections or negative results with commands and reports.

Code: [MIT](LICENSE). Downloaded models, data and grader retain their own licenses.

Signoff: @tensorhan
