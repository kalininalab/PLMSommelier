"""Dataset loading, task inference and the subsampling that makes this cheap.

* ``sequence`` is the only required column.
* the label column is ``label``, ``labels`` or ``Y`` (checked in that order),
  or an explicit ``label_col``/``--label-col``.
* the split column is ``split``, whose validation level is spelled ``valid`` by
  HuggingFace-derived datasets and ``val`` by locally split ones.

``infer_task`` only answers the questions the label column's *dtype* settles
unambiguously -- multiple columns, non-numeric values, exactly two distinct
values, or a float dtype of genuinely non-whole-numbered values (or of
whole numbers with no missing rows anywhere in the column). A whole-number
column with more than two values -- including one that pandas upcast to
float64 because of an unrelated blank row -- could be class codes or an
integer-valued regression target, and inference refuses to guess: it raises
:class:`ValueError` asking for an explicit ``task=``/``--task``.

The subsampling is the whole reason this is a tool rather than a cluster job:
15-20% of the training data is usually enough to identify the best layer.
"""

from __future__ import annotations

import heapq
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["Task", "Dataset", "load_dataset", "infer_task"]

Task = Literal["regression", "classification", "multi-label"]

_VAL_NAMES = ("valid", "val")

# A single label column is expected to hold one scalar per row. This matches
# a delimited run of >=2 numbers, optionally bracketed -- e.g. "0,1,1,0" or
# "[0, 1]" -- which is a common way people mis-encode multi-label/per-residue
# targets as a single column.
_DELIMITED_NUMERIC = re.compile(
    r"^\s*[\[\(]?\s*-?\d+(?:\.\d+)?(?:\s*[,;|\s]\s*-?\d+(?:\.\d+)?){1,}\s*[\]\)]?\s*$"
)


def _reject_structured_labels(values: pd.Series, col: str, seqs: pd.Series | None = None) -> None:
    """Raise if ``col`` actually holds list/delimited/per-residue targets.

    ``load_dataset``/``infer_task`` only recognise one scalar label per row
    in a single column; a genuinely multi-label or per-residue target encoded
    that way silently becomes an explosion of near-unique "classes" instead
    of erroring. ``seqs``, when given, enables the per-residue check (a label
    string whose length matches its row's sequence length).
    """
    non_null = values.dropna()
    if non_null.empty:
        return

    list_like = non_null[non_null.map(lambda v: isinstance(v, (list, tuple, np.ndarray)))]
    if len(list_like):
        raise ValueError(
            f"label column {col!r} holds list/array values (e.g. {list_like.iloc[0]!r}) -- "
            "plmsommelier expects one scalar label per row. For multi-label targets, use "
            "one numeric column per label and pass --label-col a b c."
        )

    str_values = non_null[non_null.map(lambda v: isinstance(v, str))]
    if str_values.empty:
        return

    delimited = str_values[str_values.map(lambda s: bool(_DELIMITED_NUMERIC.match(s)))]
    if len(delimited):
        raise ValueError(
            f"label column {col!r} holds delimited multi-value strings (e.g. "
            f"{delimited.iloc[0]!r}) -- plmsommelier expects one scalar label per row. "
            "For multi-label targets, use one numeric column per label and pass "
            "--label-col a b c."
        )

    if seqs is not None:
        aligned = seqs.reindex(str_values.index).astype(str)
        lengths_match = str_values.str.len() == aligned.str.len()
        if lengths_match.all() and (str_values.str.len() > 1).all():
            raise ValueError(
                f"label column {col!r} looks like per-residue annotations (e.g. "
                f"{str_values.iloc[0]!r}, matching its row's sequence length) -- "
                "plmsommelier only supports one label per sequence, not per residue."
            )


@dataclass(slots=True)
class Dataset:
    name: str
    task: Task
    train_seqs: list[str]
    train_y: np.ndarray
    val_seqs: list[str]
    val_y: np.ndarray
    # rows available before `sample` subsampling (after cleaning and the
    # max_seq_len cut), split by train/val. None means "unknown" -- every
    # hand-built Dataset in the test suite leaves this unset. Lets a
    # low-confidence report tell "raise --sample" apart from "you're already
    # using every row in the file".
    n_train_available: int | None = None
    n_val_available: int | None = None


def infer_task(y: np.ndarray, label_cols: list[str]) -> Task:
    """Determine the task from the label column(s). An explicit ``task=`` always wins.

    Only answers the questions the dtype settles unambiguously: multiple label
    columns, non-numeric values, exactly two distinct values, or a float
    dtype. A whole-number column with more than two values could be class
    codes or an integer-valued regression target, and there is no reliable
    signal in the values alone to tell those apart -- so this raises
    :class:`ValueError` rather than guessing.
    """
    if len(label_cols) > 1:
        return "multi-label"

    series = pd.Series(y)
    had_missing = bool(series.isna().any())
    values = series.dropna()
    n_uniq = values.nunique()

    if not pd.api.types.is_numeric_dtype(values):
        _reject_structured_labels(values, label_cols[0])
        return "classification"
    if n_uniq <= 1:
        return "regression"  # a constant numeric column can't be many classes
    if n_uniq == 2:
        return "classification"
    if pd.api.types.is_float_dtype(values):
        # pandas upcasts an otherwise-integer column to float64 the moment any
        # row is NaN, even a row dropped before this ever runs. Don't let that
        # upcast alone flip a genuinely ambiguous integer column to
        # "regression" -- only trust float dtype when nothing was missing, or
        # the values aren't actually whole numbers.
        whole_numbered = bool((values == values.round()).all())
        if not (had_missing and whole_numbered):
            return "regression"

    raise ValueError(
        f"label column {label_cols[0]!r} holds {n_uniq} distinct whole numbers, "
        "which could be class codes or an integer-valued regression target. "
        "Pass task='classification' or task='regression' (--task on the CLI) "
        "to say which."
    )


def _validate_task(task: Task, y: np.ndarray, label_cols: list[str], df: pd.DataFrame) -> None:
    """Check an explicitly-supplied ``task`` is consistent with the data."""
    if task == "multi-label":
        if len(label_cols) == 1:
            raise ValueError(
                f"task='multi-label' needs multiple label columns, got a single "
                f"column {label_cols[0]!r}"
            )
        non_numeric = [c for c in label_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"multi-label columns must be numeric, got non-numeric {non_numeric}")
        return
    if len(label_cols) > 1:
        raise ValueError(f"task={task!r} needs a single label column, got {label_cols}")
    if task == "regression" and not pd.api.types.is_numeric_dtype(pd.Series(y)):
        raise ValueError(
            f"task='regression' needs a numeric label column, {label_cols[0]!r} is not"
        )


def _encode_labels(series: pd.Series) -> dict:
    """Map a classification label column's values to 0..C-1."""
    classes = sorted(series.dropna().unique().tolist(), key=str)
    return {value: code for code, value in enumerate(classes)}


_LABEL_COL_NAMES = ("label", "labels", "Y")


def _pick_label_columns(df: pd.DataFrame, label_col: str | list[str] | None) -> list[str]:
    if isinstance(label_col, list) and label_col:
        return label_col
    if label_col:
        return [label_col]
    for name in _LABEL_COL_NAMES:
        if name in df.columns:
            return [name]
    raise ValueError(
        f"no label column found: expected {_LABEL_COL_NAMES}, or an explicit --label-col"
    )


def _resolve_sample(value: int | float, total: int) -> int:
    """Resolve ``sample`` into an absolute row count against ``total`` rows.

    A value in ``(0, 1)`` is a fraction of ``total``; a value ``>= 1`` is an
    absolute row count (rounded if it's a float, e.g. ``4000`` or ``4000.0``).
    There's no separate case for exactly ``1`` -- it's caught by the ``>= 1``
    branch and means "one row", not "100%"; pass ``0.999...`` or an explicit
    row count if 100% is what's meant.
    """
    if value <= 0:
        raise ValueError(f"sample must be > 0, got {value}")
    if value < 1:
        return max(1, round(value * total))
    return int(round(value))


def _subsample(
    df: pd.DataFrame, n: int | None, task: Task, label_cols: list[str], seed: int
) -> pd.DataFrame:
    """Cap a split at ``n`` rows, preserving class balance where meaningful."""
    if n is None or len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    if task == "classification" and len(label_cols) == 1:
        # Stratify so rare classes survive the cut; without this a heavily
        # imbalanced task loses whole classes and the probe silently degrades.
        groups = df.groupby(label_cols[0], observed=True)
        share = n / len(df)
        take = {}
        for key, grp in groups:
            take[key] = max(1, min(len(grp), round(len(grp) * share)))

        # Rounding up-then-clamping can still overshoot `n`; trim it back one
        # row at a time from whichever class currently has the most rows, so
        # the cut never empties a class that survived the initial pass.
        overshoot = sum(take.values()) - n
        heap = [(-count, key) for key, count in take.items() if count > 1]
        heapq.heapify(heap)
        while overshoot > 0 and heap:
            _, key = heapq.heappop(heap)
            take[key] -= 1
            overshoot -= 1
            if take[key] > 1:
                heapq.heappush(heap, (-take[key], key))

        picks = [
            grp.sample(n=take[key], random_state=int(rng.integers(1 << 31))) for key, grp in groups
        ]
        return pd.concat(picks)
    return df.sample(n=n, random_state=seed)


def load_dataset(
    path: str | Path,
    *,
    task: Task | None = None,
    label_col: str | list[str] | None = None,
    sample: int | float | None = 5000,
    max_seq_len: int | None = 2000,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> Dataset:
    """Load a CSV, infer the task, split it, and subsample to the budget.

    ``max_seq_len`` drops sequences longer than this many residues before the
    split and the subsample -- not after -- so a single pathological outlier
    (a 13k- or 35k-residue protein has turned up in real datasets) can never
    survive subsampling into a run. Attention memory is quadratic in sequence
    length, so one such sequence forming its own batch can exceed a GPU's
    budget regardless of how conservative ``max_tokens_per_batch`` is set
    elsewhere. Pass ``None`` (or ``0``) to disable the cap.

    ``sample`` caps the whole (post-cleaning, pre-split) dataset, then
    ``val_fraction`` divides that budget between train and validation the
    same way it divides the full dataset when no cap applies. A value
    ``>= 1`` is an absolute row count (e.g. ``5000``); a value in ``(0, 1)``
    is a fraction of the whole dataset, e.g. ``0.1`` for 10%. Pass ``None``
    to disable subsampling entirely.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "sequence" not in df.columns:
        raise ValueError(f"{path}: missing required column 'sequence'")

    label_cols = _pick_label_columns(df, label_col)
    df = df.dropna(subset=["sequence"])

    if max_seq_len:
        lengths = df["sequence"].astype(str).str.len()
        too_long = lengths > max_seq_len
        n_dropped = int(too_long.sum())
        if n_dropped:
            warnings.warn(
                f"{path}: dropping {n_dropped} sequence(s) longer than "
                f"max_seq_len={max_seq_len} (longest seen: {int(lengths.max())} "
                "residues) -- attention memory is quadratic in length, so a single "
                "outlier can exceed a batch's memory budget regardless of "
                "max_tokens_per_batch.",
                RuntimeWarning,
                stacklevel=2,
            )
        df = df[~too_long]
        if df.empty:
            raise ValueError(f"{path}: every sequence is longer than max_seq_len={max_seq_len}")

    if len(label_cols) == 1:
        _reject_structured_labels(df[label_cols[0]], label_cols[0], df["sequence"])

    # Task inference sees the label column *with* its missing rows still in
    # it -- dropping them first would erase the very NaN-driven dtype upcast
    # (an integer column becomes float64 the moment any row is blank) that
    # inference needs to see through. The dropped rows never reach `build()`
    # below, since that works off `df` after this dropna.
    inference_y = (
        df[label_cols[0]].to_numpy() if len(label_cols) == 1 else df[label_cols].to_numpy()
    )
    if task is not None:
        _validate_task(task, inference_y, label_cols, df)
        resolved_task: Task = task
    else:
        resolved_task = infer_task(inference_y, label_cols)

    df = df.dropna(subset=label_cols)

    resolved_sample = _resolve_sample(sample, len(df)) if sample is not None else None
    if resolved_sample is None:
        train_cap, val_cap = None, None
    else:
        val_cap = max(1, round(resolved_sample * val_fraction))
        train_cap = max(1, resolved_sample - val_cap)

    # --- splits --------------------------------------------------------
    if "split" in df.columns:
        present = set(df["split"].astype(str).unique())
        val_name = next((v for v in _VAL_NAMES if v in present), None)
        if val_name is None:
            raise ValueError(f"{path}: 'split' column has no {_VAL_NAMES} level")
        train_df = df[df["split"].astype(str) == "train"]
        val_df = df[df["split"].astype(str) == val_name]
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"{path}: train or {val_name} split is empty")
    else:
        shuffled = df.sample(frac=1.0, random_state=seed)
        cut = int(len(shuffled) * (1 - val_fraction))
        train_df, val_df = shuffled.iloc[:cut], shuffled.iloc[cut:]

    n_train_available, n_val_available = len(train_df), len(val_df)
    train_df = _subsample(train_df, train_cap, resolved_task, label_cols, seed)
    val_df = _subsample(val_df, val_cap, resolved_task, label_cols, seed + 1)

    code_of = {}
    if resolved_task == "classification":
        # Coded over the full (pre-split) column so train and val share one
        # code space even if a rare class only survives in one of them.
        code_of = _encode_labels(df[label_cols[0]])

    def build(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        if resolved_task == "classification":
            y = frame[label_cols[0]].map(code_of).to_numpy(dtype=int)
        elif len(label_cols) > 1:
            y = frame[label_cols].to_numpy(dtype=float)
        else:
            y = frame[label_cols[0]].to_numpy(dtype=float)
        return frame["sequence"].astype(str).tolist(), y

    train_seqs, train_y = build(train_df)
    val_seqs, val_y = build(val_df)

    return Dataset(
        name=path.stem,
        task=resolved_task,
        train_seqs=train_seqs,
        train_y=train_y,
        val_seqs=val_seqs,
        val_y=val_y,
        n_train_available=n_train_available,
        n_val_available=n_val_available,
    )
