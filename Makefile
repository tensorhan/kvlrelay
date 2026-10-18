# Thin recipes over the Python CLIs. No training or downloads on bare `make`.
.DEFAULT_GOAL := help
.NOTPARALLEL:
.PHONY: help prepare train benchmark ifeval ifeval-native run recipe

PYTHON ?= .venv/bin/python
PROFILE ?= pilot
DEVICE ?= auto
DTYPE ?= bfloat16
SEED ?= 42
DATA_SEED ?= 42
LR ?= 1e-3
DATASET_REVISION := 1a9454ed054b8544503ab8d8c0a519d141a44c5b
SOURCE_REVISION := c1899de289a04d12100db370d81485cdf75e47ca
TARGET_REVISION := 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

ifeq ($(PROFILE),pilot)
TRAIN_SAMPLES ?= 256
FIT_SAMPLES ?= 256
STEPS ?= 100
BATCH ?= 1
PREFIX ?= 1024
NATURAL_MAX ?= 768
PACKED_MIN ?= 768
CONTINUATION ?= 128
MAX_SCAN ?= 20000
IFEVAL_LIMIT ?= 32
else ifeq ($(PROFILE),paper-budget)
# One seed/rate of the head-wise recipe, not the complete paper experiment.
# 40,000 training examples is our choice for 5,000 x 8 example exposures.
TRAIN_SAMPLES ?= 40000
FIT_SAMPLES ?= 400
STEPS ?= 5000
BATCH ?= 8
PREFIX ?= 16384
NATURAL_MAX ?= 10000
PACKED_MIN ?= 12000
CONTINUATION ?= 2048
MAX_SCAN ?= 1000000
IFEVAL_LIMIT ?= 541
else
$(error Unknown PROFILE '$(PROFILE)'; choose pilot or paper-budget)
endif

DATA_DIR ?= data/$(PROFILE)-data$(DATA_SEED)
RUN ?= $(PROFILE)-data$(DATA_SEED)-seed$(SEED)-lr$(LR)-$(DEVICE)-$(DTYPE)
ARTIFACT ?= artifacts/$(RUN)/translator.pt
RESULTS ?= results/$(RUN)
PAIR = --source-revision $(SOURCE_REVISION) --target-revision $(TARGET_REVISION) \
	--device $(DEVICE) --dtype $(DTYPE) --seed $(SEED)
# Optional CLI arguments, e.g. IFEVAL_ARGS=--resume or a timing context ladder.
BENCHMARK_ARGS ?=
IFEVAL_ARGS ?=
TRAIN_ARGS ?=
RECIPE_ARGS ?=

help:
	@echo "Targets: prepare, train, benchmark, ifeval, ifeval-native, run, recipe"
	@echo "Defaults: PROFILE=pilot DEVICE=auto DTYPE=bfloat16; inspect with make -n run"
	@echo "Larger recipe: make -n run PROFILE=paper-budget DEVICE=cuda"
	@echo "Use RUN=name for a fresh experiment; generated reports stay in results/"

prepare:
	$(PYTHON) paper_data.py nemotron --dataset-revision $(DATASET_REVISION) \
		--tokenizer-revision $(TARGET_REVISION) --mixture mixed --reasoning both \
		--train-samples $(TRAIN_SAMPLES) --validation-samples 64 --test-samples 64 \
		--natural-max-tokens $(NATURAL_MAX) --packed-min-tokens $(PACKED_MIN) \
		--max-prefix-tokens $(PREFIX) --continuation-tokens $(CONTINUATION) \
		--max-scan $(MAX_SCAN) --seed $(DATA_SEED) --output $(DATA_DIR)
	$(PYTHON) ifeval.py ifeval --prepare

train:
	$(PYTHON) train_translator.py --data-dir $(DATA_DIR) $(PAIR) \
		--fit-samples $(FIT_SAMPLES) --distill-steps $(STEPS) --batch-size $(BATCH) \
		--continuation-tokens $(CONTINUATION) --learning-rate $(LR) --output $(ARTIFACT) $(TRAIN_ARGS)

benchmark:
	$(PYTHON) benchmark.py nemotron --data-dir $(DATA_DIR) --translator $(ARTIFACT) $(PAIR) \
		--continuation-tokens $(CONTINUATION) --repeats 20 --warmup 3 --timing-examples 8 \
		--output $(RESULTS)/nemotron.json $(BENCHMARK_ARGS)

ifeval:
	$(PYTHON) ifeval.py ifeval --translator $(ARTIFACT) $(PAIR) --limit $(IFEVAL_LIMIT) \
		--output $(RESULTS)/ifeval.json $(IFEVAL_ARGS)

# Full 541-prompt native baseline, independent of training and profile subset limits.
ifeval-native:
	$(PYTHON) ifeval.py ifeval --native-only $(PAIR) \
		--output $(RESULTS)/ifeval-native.json $(IFEVAL_ARGS)

# Consumes prepared data. Five-rate selection (extended at edges), then seeds 42/43/44.
recipe:
	$(PYTHON) recipe.py $(PAIR) --data-dir $(DATA_DIR) \
		--fit-samples $(FIT_SAMPLES) --distill-steps $(STEPS) --batch-size $(BATCH) \
		--continuation-tokens $(CONTINUATION) --lr-center $(LR) --ifeval-limit $(IFEVAL_LIMIT) \
		--artifact-dir artifacts/$(RUN)/recipe --output-dir $(RESULTS)/recipe $(RECIPE_ARGS)

# Always collect task quality even when the numerical/timing screen fails.
# Keep preparation/training failures fatal; propagate evaluation failures last.
run:
	$(MAKE) prepare
	$(MAKE) train
	@status=0; $(MAKE) benchmark || status=$$?; \
		$(MAKE) ifeval || status=$$?; exit $$status
