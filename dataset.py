"""Composition-disjoint MIDI datasets and random next-token batches."""

from bisect import bisect_right
from itertools import accumulate, islice
import random
import unicodedata
import warnings

import torch
from torch.utils.data import Dataset, Sampler
from symusic import Score

from tokenizer import TokenizerConfig, create_event, define_vocab, preprocess_performance, tokenize


TARGET_TPQ = 480


def composition_key(row):
    """Keep all movements and performances of a named work in one split."""
    def normalize(field):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Performance {row.get('id')} lacks {field}")
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    return normalize("composer"), normalize("composition")


def split_by_composition(rows, validation_fraction=0.2, seed=42):
    """Split metadata before tokenization/windowing, using whole compositions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    rows = list(rows)
    keys = sorted({composition_key(row) for row in rows})
    if len(keys) < 2:
        raise ValueError("Need at least two compositions; load more performances")
    random.Random(seed).shuffle(keys)
    count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:count])
    train, validation = [], []
    for row in rows:
        target = validation if composition_key(row) in validation_keys else train
        target.append(row)
    return train, validation


def select_performances(rows, num_compositions=32, performances_per_composition=2,
                        max_performances=64, seed=42):
    """Select distinct works first, then round-robin capped performances per work."""
    if num_compositions < 2 or performances_per_composition < 1:
        raise ValueError("Need at least two compositions and one performance per composition")
    if max_performances < num_compositions:
        raise ValueError("max_performances must be at least num_compositions")
    groups, seen_ids = {}, set()
    for row in rows:
        if not row["tier_b"]:
            continue
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate performance ID: {row['id']}")
        seen_ids.add(row["id"])
        groups.setdefault(composition_key(row), []).append(row)
    if len(groups) < num_compositions:
        raise ValueError(f"Requested {num_compositions} compositions, found {len(groups)}; "
                         "increase max_scan or reduce num_compositions")
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys = keys[:num_compositions]
    for key in keys:
        groups[key].sort(key=lambda row: str(row["id"]))
        rng.shuffle(groups[key])
    selected = []
    for index in range(performances_per_composition):
        for key in keys:
            if index < len(groups[key]):
                selected.append(groups[key][index])
                if len(selected) == max_performances:
                    return selected
    return selected


def select_with_holdout(rows, heldout_performances, num_compositions,
                        performances_per_composition=2, max_performances=64, seed=42):
    """Select training works while reserving every composition in the saved holdout.

    The composition and performance budgets include the saved validation set.
    Other performances of held-out works are excluded, even with different IDs.
    """
    heldout_keys = {tuple(p["composition_key"]) for p in heldout_performances}
    heldout_ids = {p["id"] for p in heldout_performances}
    if not heldout_keys or len(heldout_ids) != len(heldout_performances):
        raise ValueError("Saved holdout must contain nonempty, unique performance IDs")
    train_count = num_compositions - len(heldout_keys)
    if train_count < 2:
        raise ValueError("Composition budget must leave at least two training compositions")
    candidates = [row for row in rows if row["tier_b"] and composition_key(row) not in heldout_keys
                  and str(row["id"]) not in heldout_ids]
    return select_performances(candidates, train_count, performances_per_composition,
                               max_performances - len(heldout_performances), seed)


def load_performances(max_performances=64, num_compositions=32,
                      performances_per_composition=2, max_scan=5000, seed=42,
                      heldout_performances=None):
    """Scan bounded metadata first; fetch MIDI payloads only for selected works."""
    from datasets import load_dataset

    if max_scan < num_compositions:
        raise ValueError("max_scan must be at least num_compositions")
    source = load_dataset(
        "SyMuPe/PianoCoRe", split="train", streaming=True,
        columns=["id", "composer", "composition", "tier_b"],
    )
    metadata = list(islice(source, max_scan))
    if heldout_performances is None:
        selected = select_performances(metadata, num_compositions,
                                       performances_per_composition, max_performances, seed)
    else:
        selected = select_with_holdout(metadata, heldout_performances, num_compositions,
                                        performances_per_composition, max_performances, seed)
    ids = [row["id"] for row in selected]
    if heldout_performances is not None:
        ids.extend(p["id"] for p in heldout_performances)
    print(f"Selected {len(ids)} performances across {num_compositions} compositions "
          f"from {len(metadata)} metadata rows", flush=True)
    return load_performances_by_ids(ids)


def load_performances_by_ids(ids):
    """Fetch exactly the saved performance IDs, preserving their saved order."""
    from datasets import load_dataset

    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Performance IDs must be nonempty and unique")
    payloads = load_dataset(
        "SyMuPe/PianoCoRe", split="train", streaming=True,
        filters=[("id", "in", ids)],
        columns=["id", "composer", "composition", "tier_b", "performance_midi_bytes"],
    )
    fetched = {}
    for row in payloads:
        if row["id"] in fetched:
            raise ValueError(f"Duplicate performance ID: {row['id']}")
        fetched[row["id"]] = row
        if len(fetched) == len(ids):
            break
    if set(fetched) != set(ids):
        raise ValueError("Missing MIDI payloads for selected performances")
    return [fetched[sample_id] for sample_id in ids]


class PerformanceWindowDataset(Dataset):
    """All complete context_length+1 windows, without joining performances.

    Store each full token sequence once; indices address slices of that sequence.
    Short performances are reported and skipped, never padded or concatenated.
    """

    def __init__(self, rows, context_length=256, target_tpq=TARGET_TPQ, tokenizer_config=None):
        if context_length < 1 or target_tpq < 1:
            raise ValueError("context_length and target_tpq must be positive")
        self.context_length = context_length
        self.target_tpq = target_tpq
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.vocabulary, self.token_to_id, _ = define_vocab(self.tokenizer_config)
        self.sequences = []
        self.performance_ids = []
        self.composition_keys = []
        self.skipped_ids = []
        seen_ids = set()
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate performance ID: {sample_id}")
            seen_ids.add(sample_id)
            key = composition_key(row)
            score = Score.from_midi(row["performance_midi_bytes"])
            notes = preprocess_performance(score, target_tpq=target_tpq,
                                           time_step=self.tokenizer_config.time_step)
            _, ids = tokenize(self.token_to_id, create_event(notes), self.tokenizer_config)
            # tokenize already appends the true end marker. Add BOS only here,
            # before taking any windows, at the start of the full performance.
            ids = [self.token_to_id["BOS"], *ids]
            if len(ids) < context_length + 1:
                self.skipped_ids.append(sample_id)
                warnings.warn(f"Skipping short performance {sample_id}: {len(ids)} tokens")
                continue
            self.sequences.append(torch.tensor(ids, dtype=torch.long))
            self.performance_ids.append(sample_id)
            self.composition_keys.append(key)
        if not self.sequences:
            raise ValueError("No performances long enough for a complete window")
        self.window_counts = [len(ids) - context_length for ids in self.sequences]
        self.window_ends = list(accumulate(self.window_counts))
        self.window_starts = [0, *self.window_ends[:-1]]

    def __len__(self):
        return self.window_ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        performance = bisect_right(self.window_ends, index)
        offset = index - self.window_starts[performance]
        window = self.sequences[performance][offset:offset + self.context_length + 1]
        return window[:-1], window[1:]


class RandomWindowBatchSampler(Sampler):
    """Sample a performance uniformly, then a valid start uniformly, per row.

    Sampling is with replacement, so long works don't dominate and even small
    collections produce full batches. A private RNG advances across iterations.
    """

    def __init__(self, dataset, num_batches, batch_size=8, seed=42):
        if num_batches < 1 or batch_size < 1:
            raise ValueError("num_batches and batch_size must be positive")
        self.dataset = dataset
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            performances = torch.randint(
                len(self.dataset.sequences), (self.batch_size,), generator=self.generator
            ).tolist()
            indices = []
            for performance in performances:
                offset = torch.randint(
                    self.dataset.window_counts[performance], (1,), generator=self.generator
                ).item()
                indices.append(self.dataset.window_starts[performance] + offset)
            yield indices


def build_datasets(rows, context_length=256, target_tpq=TARGET_TPQ,
                   validation_fraction=0.2, seed=42, tokenizer_config=None):
    rows = list(rows)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate performance IDs in collection")
    train_rows, validation_rows = split_by_composition(rows, validation_fraction, seed)
    return (
        PerformanceWindowDataset(train_rows, context_length, target_tpq, tokenizer_config),
        PerformanceWindowDataset(validation_rows, context_length, target_tpq, tokenizer_config),
    )
