"""Teacher-forced continuation metrics, matching the paper's Nemotron LM row."""

import math
import statistics

import torch
import torch.nn.functional as F


def example_tensors(example, adapter, steps):
    prompt = torch.tensor([example["prompt_ids"]], device=adapter.device, dtype=torch.long)
    labels = torch.tensor(example["continuation_ids"][:steps], device=adapter.device, dtype=torch.long)
    if prompt.shape[1] < 2 or labels.numel() < 1:
        raise ValueError("Need a prompt of at least two tokens and a nonempty continuation")
    if prompt.shape[1] + labels.numel() - 1 > adapter.config.max_position_embeddings:
        raise ValueError("Prompt plus scored continuation exceeds the context window")
    return prompt, labels


def continuation_logits(adapter, prompt, labels, cache=None):
    """Return one distribution per gold token; each gold input is consumed once."""
    start = 0 if cache is None else prompt.shape[1] - 1
    initial = prompt if cache is None else prompt[:, -1:]
    output = adapter.forward(initial, start=start, cache=cache)
    pieces = [output.logits[:, -1:, :]]
    if labels.numel() > 1:
        tail = adapter.forward(labels[:-1].unsqueeze(0), start=prompt.shape[1],
                               cache=output.past_key_values, all_logits=True)
        pieces.append(tail.logits)
        output = tail
    return torch.cat(pieces, dim=1), int(output.past_key_values.get_seq_length())


def distillation_loss(translator, example, steps):
    source, target = translator.source, translator.target
    prompt, labels = example_tensors(example, target, steps)
    with torch.no_grad():
        features = source.capture(prompt[:, :-1].to(source.device))
        teacher, _ = continuation_logits(target, prompt, labels)
    student, _ = continuation_logits(target, prompt, labels, translator.cache(features))
    return F.kl_div(student.float().log_softmax(-1), teacher.float().log_softmax(-1),
                    log_target=True, reduction="batchmean") / labels.numel()


def distribution_metrics(reference, actual, labels):
    # Keep the high-precision metrics on CPU, including when logits are on MPS.
    reference, actual = reference.detach().cpu().double(), actual.detach().cpu().double()
    labels = labels.detach().cpu().reshape(1, -1, 1)
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise RuntimeError("Nonfinite logits in handoff evaluation")
    if reference.shape != actual.shape or reference.shape[1] != labels.shape[1]:
        raise ValueError("Logit/label alignment mismatch")
    logp, logq = reference.log_softmax(-1), actual.log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1).mean().clamp_min(0)
    nll = -logq.gather(-1, labels).sum()
    native_nll = -logp.gather(-1, labels).sum()
    agreement = (reference.argmax(-1) == actual.argmax(-1)).double().mean()
    return {"mean_kl_nats": float(kl), "max_abs_logit_error": float((reference - actual).abs().max()),
            "token_agreement": float(agreement), "scored_tokens": labels.numel(),
            "nll_sum_nats": float(nll), "native_nll_sum_nats": float(native_nll),
            "delta_log_perplexity_nats": float((nll - native_nll) / labels.numel())}


def summarize_quality(items):
    count = sum(item["scored_tokens"] for item in items)
    if not items or not count:
        raise ValueError("No quality observations")
    mean_nll = sum(item["nll_sum_nats"] for item in items) / count
    native_nll = sum(item["native_nll_sum_nats"] for item in items) / count
    kls = [item["mean_kl_nats"] for item in items]
    return {"examples": len(items), "scored_tokens": count,
            "mean_kl_nats": statistics.mean(kls),
            "kl_standard_error_over_conversations": statistics.stdev(kls) / math.sqrt(len(kls)) if len(kls) > 1 else None,
            "token_weighted_kl_nats": sum(item["mean_kl_nats"] * item["scored_tokens"] for item in items) / count,
            "token_agreement": sum(item["token_agreement"] * item["scored_tokens"] for item in items) / count,
            "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
            "native_target_perplexity": math.exp(native_nll) if native_nll < 700 else None,
            "delta_log_perplexity_nats": mean_nll - native_nll}


@torch.no_grad()
def evaluate(translator, examples, steps):
    items = []
    for example in examples:
        prompt, labels = example_tensors(example, translator.target, steps)
        reference, _ = continuation_logits(translator.target, prompt, labels)
        features = translator.source.capture(prompt[:, :-1].to(translator.source.device))
        actual, length = continuation_logits(translator.target, prompt, labels, translator.cache(features))
        if length != prompt.shape[1] + labels.numel() - 1:
            raise RuntimeError("Incorrect translated-cache sequence length")
        items.append(distribution_metrics(reference, actual, labels))
    return summarize_quality(items)
