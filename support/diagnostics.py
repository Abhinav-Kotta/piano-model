"""Token-family diagnostics on fixed, teacher-forced validation windows."""

import math

import torch
from torch.nn import functional as F
from support.event_types import event_type_ids


GROUPS = ("note_on", "note_off", "velocity", "time_shift")


def fit_baselines(dataset):
    """Fit modal token values using training sequences only."""
    counts = torch.zeros(len(dataset.vocabulary), dtype=torch.long)
    for sequence in dataset.sequences:
        counts += torch.bincount(sequence, minlength=len(counts))
    return {
        group: max((i for i, token in enumerate(dataset.vocabulary)
                    if token.startswith(group.upper() + "_")), key=lambda i: int(counts[i]))
        for group in ("velocity", "time_shift")
    }


@torch.no_grad()
def evaluate(model, batches, device, vocabulary=None, baselines=None):
    was_training = model.training
    model.eval()
    totals = dict(loss=0.0, correct=0, top5=0, tokens=0, type_loss=0.0, type_correct=0)
    stats = [dict(tokens=0, correct=0, top5=0, loss=0.0, right_type=0,
                  within_one=0, absolute_error=0.0, mode=0, previous=0,
                  type_loss=0.0, conditional_loss=0.0, conditional_correct=0,
                  marginal_type_correct=0) for _ in GROUPS]
    if vocabulary is not None:
        group_ids = [next((i for i, group in enumerate(GROUPS)
                          if token.startswith(group.upper() + "_")), -1) for token in vocabulary]
        token_groups = torch.tensor(group_ids, device=device)
        values = [int(token.rsplit("_", 1)[1]) if group_ids[i] >= 0 else 0
                  for i, token in enumerate(vocabulary)]
        ranks = [0] * len(vocabulary)
        for group in range(4):
            for rank, token_id in enumerate(sorted(
                    (i for i, g in enumerate(group_ids) if g == group), key=lambda i: values[i])):
                ranks[token_id] = rank
        token_ranks = torch.tensor(ranks, device=device)
        token_values = torch.tensor(values, device=device)
        families = torch.tensor(event_type_ids(vocabulary), device=device)
        family_indices = [torch.where(families == i)[0] for i in range(7)]
    try:
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").reshape_as(y)
            predicted = logits.argmax(-1)
            matches = predicted == y
            top5 = (logits.topk(min(5, logits.shape[-1]), dim=-1).indices == y.unsqueeze(-1)).any(-1)
            totals["loss"] += losses.sum().item()
            totals["correct"] += matches.sum().item()
            totals["top5"] += top5.sum().item()
            totals["tokens"] += y.numel()
            if vocabulary is None:
                continue
            target_groups, predicted_groups = token_groups[y], token_groups[predicted]
            log_prob = logits.log_softmax(-1)
            # Summed family probability, not the family of the best single token.
            type_log_prob = torch.stack([log_prob.index_select(-1, ids).logsumexp(-1)
                                         for ids in family_indices], dim=-1)
            type_losses = -type_log_prob.gather(-1, families[y].unsqueeze(-1)).squeeze(-1)
            type_matches = type_log_prob.argmax(-1) == families[y]
            totals["type_loss"] += type_losses.sum().item()
            totals["type_correct"] += type_matches.sum().item()
            for i, group in enumerate(GROUPS):
                mask = target_groups == i
                same_type = mask & (predicted_groups == i)
                entry = stats[i]
                entry["tokens"] += mask.sum().item()
                entry["correct"] += (mask & matches).sum().item()
                entry["top5"] += (mask & top5).sum().item()
                entry["loss"] += losses[mask].sum().item()
                entry["type_loss"] += type_losses[mask].sum().item()
                entry["conditional_loss"] += (losses - type_losses)[mask].sum().item()
                entry["marginal_type_correct"] += (mask & type_matches).sum().item()
                ids = family_indices[i + 3]
                if len(ids):
                    conditional_prediction = ids[logits.index_select(-1, ids).argmax(-1)]
                    entry["conditional_correct"] += (mask & (conditional_prediction == y)).sum().item()
                entry["right_type"] += same_type.sum().item()
                entry["within_one"] += (same_type & ((token_ranks[predicted] - token_ranks[y]).abs() <= 1)).sum().item()
                entry["absolute_error"] += (token_values[predicted] - token_values[y]).abs()[same_type].sum().item()
                if baselines and group in baselines:
                    entry["mode"] += (mask & (y == baselines[group])).sum().item()
            if baselines:
                # x[:, t] is observed before predicting y[:, t]. No future labels.
                positions = torch.arange(1, x.shape[1] + 1, device=device).expand_as(x)
                last_position = torch.where(token_groups[x] == 2, positions, 0).cummax(dim=1).values
                previous = x.gather(1, (last_position - 1).clamp_min(0))
                previous = torch.where(last_position > 0, previous, baselines["velocity"])
                stats[2]["previous"] += ((target_groups == 2) & (y == previous)).sum().item()
    finally:
        model.train(was_training)
    if not totals["tokens"]:
        raise ValueError("Validation batches contain no targets")
    loss = totals["loss"] / totals["tokens"]
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    metrics = {"val_loss": loss, "val_accuracy": totals["correct"] / totals["tokens"],
               "val_top5_accuracy": totals["top5"] / totals["tokens"]}
    if vocabulary is not None:
        metrics.update(val_type_loss=totals["type_loss"] / totals["tokens"],
                       val_value_loss_given_true_type=loss - totals["type_loss"] / totals["tokens"],
                       val_marginal_type_accuracy=totals["type_correct"] / totals["tokens"])
    for group, entry in zip(GROUPS, stats):
        count = entry["tokens"]
        if not count:
            continue
        prefix = f"val_{group}"
        metrics.update({f"{prefix}_accuracy": entry["correct"] / count,
                        f"{prefix}_tokens": count,
                        f"{prefix}_type_loss": entry["type_loss"] / count,
                        f"{prefix}_value_loss_given_true_type": entry["conditional_loss"] / count,
                        f"{prefix}_value_accuracy_given_true_type": entry["conditional_correct"] / count,
                        f"{prefix}_marginal_type_accuracy": entry["marginal_type_correct"] / count})
        if group not in ("velocity", "time_shift"):
            continue
        metrics.update({f"{prefix}_loss": entry["loss"] / count,
                        f"{prefix}_top5_accuracy": entry["top5"] / count,
                        f"{prefix}_type_accuracy": entry["right_type"] / count,
                        f"{prefix}_within_one_bin_accuracy": entry["within_one"] / count})
        if entry["right_type"]:
            metrics[f"{prefix}_mae_given_type"] = entry["absolute_error"] / entry["right_type"]
        if baselines:
            metrics[f"{prefix}_mode_baseline_accuracy"] = entry["mode"] / count
            if group == "velocity":
                metrics[f"{prefix}_previous_baseline_accuracy"] = entry["previous"] / count
    return metrics
