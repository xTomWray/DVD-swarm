"""Run-level train/val/test split with class-presence guarantees.

Shared by ``biLSTM-1Layer-Protocol.py`` and ``biLSTM-2Layer-Protocol.py``.

Two problems the bare stratified split has on small datasets:

1. With only 3 benign runs total, ``train_test_split(..., stratify=labels,
   test_size=0.2)`` can round the val/test slice to zero of the minority class.
   That produces a val set with class-zero support, which makes the threshold-
   selection step at ``precision_recall_curve`` meaningless and trashes test
   metrics downstream.
2. With < 3 runs of a class, no balanced run-level split is possible at all.

This module:

- Hard-fails when either class has < 3 runs (override with ``allow_imbalanced=True``).
- Runs the same nested stratified split the trainer already uses.
- *Patches* the result so each of val and test contains at least one run of each
  class observed in the dataset, swapping deterministically chosen train runs
  into the deficient split when needed.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RunSplit:
    train_runs: list[str]
    val_runs: list[str]
    test_runs: list[str]
    notes: list[str] = field(default_factory=list)


def _stratified_split_with_fallback(
    items: list[str],
    labels_of: dict[str, str],
    test_size: float,
    label: str,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Same semantics as the original _safe_run_split in the trainers.

    Stratify only when there are 2+ distinct labels and every label has 2+
    members; otherwise fall back to unstratified random.
    """
    strat = [labels_of[r] for r in items]
    counts = Counter(strat)
    use_stratify = len(counts) > 1 and min(counts.values()) >= 2
    try:
        train, test = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            stratify=strat if use_stratify else None,
        )
    except ValueError as exc:
        print(f"  WARN: stratified {label} split failed ({exc}); falling back to unstratified.")
        train, test = train_test_split(items, test_size=test_size, random_state=seed)
    return list(train), list(test)


def _classes_in(runs: Iterable[str], labels_of: dict[str, str]) -> set[str]:
    return {labels_of[r] for r in runs}


def _patch_quota(
    train_runs: list[str],
    target_runs: list[str],
    target_name: str,
    expected_classes: set[str],
    labels_of: dict[str, str],
    notes: list[str],
) -> tuple[list[str], list[str]]:
    """Ensure ``target_runs`` contains at least one run of each expected class.

    Mutates copies of the input lists and returns them. Each missing class is
    filled by moving the deterministically-first eligible run out of train.
    """
    train_runs = list(train_runs)
    target_runs = list(target_runs)
    present = _classes_in(target_runs, labels_of)
    missing = sorted(expected_classes - present)
    for cls in missing:
        candidates = sorted(r for r in train_runs if labels_of[r] == cls)
        if not candidates:
            notes.append(
                f"could not patch {target_name}: no {cls} runs available in train "
                f"(should not happen given the ≥3-per-class precondition)"
            )
            continue
        picked = candidates[0]
        train_runs.remove(picked)
        target_runs.append(picked)
        notes.append(
            f"patched: moved {os.path.basename(picked)} from train to {target_name} "
            f"to add {cls} class"
        )
    return train_runs, target_runs


def run_level_three_way_split(
    run_paths: list[str],
    run_labels: dict[str, str],
    *,
    test_frac: float,
    val_frac: float,
    seed: int = 42,
    allow_imbalanced: bool = False,
) -> RunSplit:
    """Nested stratified split with class-presence quotas in val and test.

    Args:
        run_paths: All eligible run directories.
        run_labels: ``{run_path: 'benign' | 'attack'}``. Must cover every entry
            in ``run_paths``.
        test_frac: Fraction of ``run_paths`` reserved for held-out test.
        val_frac: Fraction of the post-test "trainval" pool reserved for val.
        seed: Random seed for reproducibility.
        allow_imbalanced: When False (default), raise ``RuntimeError`` if any
            class has fewer than 3 runs. Set True for experimentation only.

    Returns:
        A ``RunSplit`` whose ``notes`` field lists any quota-patches applied.
    """
    if not run_paths:
        raise RuntimeError("run_level_three_way_split: empty run_paths")
    missing_labels = [r for r in run_paths if r not in run_labels]
    if missing_labels:
        raise RuntimeError(
            f"run_level_three_way_split: {len(missing_labels)} run(s) missing labels: "
            f"{missing_labels[:3]}"
        )

    class_counts = Counter(run_labels[r] for r in run_paths)
    observed_classes = set(class_counts)

    if not allow_imbalanced:
        for cls in ("benign", "attack"):
            n = class_counts.get(cls, 0)
            if n < 3:
                raise RuntimeError(
                    f"{cls} class has {n} run(s) — need ≥3 for run-level train/val/test. "
                    f"Run `make audit DATA=<dir>` for the full picture, then `make sim-benign` "
                    f"or `make sim-multi` to collect more, or pass --allow-imbalanced to override."
                )

    notes: list[str] = []

    trainval_runs, test_runs = _stratified_split_with_fallback(
        run_paths, run_labels, test_frac, "test", seed
    )
    train_runs, val_runs = _stratified_split_with_fallback(
        trainval_runs, run_labels, val_frac, "val", seed
    )

    # Quota patches: only enforce membership of the classes that actually
    # exist in the dataset. Otherwise a single-class dataset (allow_imbalanced)
    # would fail the patch.
    train_runs, val_runs = _patch_quota(
        train_runs, val_runs, "val", observed_classes, run_labels, notes
    )
    train_runs, test_runs = _patch_quota(
        train_runs, test_runs, "test", observed_classes, run_labels, notes
    )

    train_set, val_set, test_set = set(train_runs), set(val_runs), set(test_runs)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError(
            "Run-level split overlap detected after quota patch — internal bug"
        )

    return RunSplit(
        train_runs=sorted(train_runs),
        val_runs=sorted(val_runs),
        test_runs=sorted(test_runs),
        notes=notes,
    )
