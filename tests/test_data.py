"""Dataset loading, task inference and subsampling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from plmsommelier.data import infer_task, load_dataset


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _frame(n=200, **cols):
    base = {
        "ID": [f"P{i:05d}" for i in range(n)],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY"[: 6 + i % 12] for i in range(n)],
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestInferTask:
    def test_continuous_is_regression(self):
        rng = np.random.default_rng(0)
        assert infer_task(rng.normal(size=500), ["label"]) == "regression"

    def test_two_values_is_classification(self):
        assert infer_task(np.array([0, 1] * 50), ["label"]) == "classification"

    def test_dense_integer_codes_are_ambiguous(self):
        """Integer label columns with >2 values could be class codes or an
        integer-valued regression target -- inference refuses to guess and
        names the offending column instead."""
        with pytest.raises(ValueError, match="class_"):
            infer_task(np.repeat(np.arange(7), 30), ["class_"])

    def test_identifier_like_column_is_ambiguous(self):
        with pytest.raises(ValueError):
            infer_task(np.arange(500), ["ID"])

    def test_several_columns_is_multi_label(self):
        assert infer_task(np.zeros((10, 3)), ["a", "b", "c"]) == "multi-label"

    def test_string_labels_are_classification(self):
        labels = np.array(["alpha", "beta", "gamma"] * 10)
        assert infer_task(labels, ["label"]) == "classification"

    def test_bool_column_is_classification(self):
        assert infer_task(np.array([True, False] * 10), ["label"]) == "classification"

    def test_non_integral_floats_are_regression(self):
        assert infer_task(np.array([1.1, 2.2, 3.3] * 10), ["label"]) == "regression"

    def test_whole_number_floats_with_no_missing_are_regression(self):
        """The deliberate boundary: a float dtype of whole numbers is still
        trusted as regression when nothing in the column is missing."""
        assert infer_task(np.repeat(np.arange(7), 30).astype(float), ["label"]) == "regression"

    def test_whole_number_floats_with_a_missing_row_are_ambiguous(self):
        """Regression test: pandas upcasts an otherwise-integer column to
        float64 the moment any row is NaN. That upcast alone must not flip
        inference from 'ambiguous, ask the user' to 'guess regression'."""
        y = np.repeat(np.arange(7), 30).astype(float)
        y[0] = np.nan
        with pytest.raises(ValueError, match="class_"):
            infer_task(y, ["class_"])

    def test_non_integral_floats_with_a_missing_row_are_still_regression(self):
        y = np.array([1.1, 2.2, 3.3] * 10)
        y[0] = np.nan
        assert infer_task(y, ["label"]) == "regression"

    def test_constant_non_numeric_column_is_classification_not_a_crash(self):
        """Regression test: the n_uniq<=1 shortcut used to fire before the
        numeric-dtype check, so a constant string column was labeled
        'regression' and then crashed downstream trying to cast it to
        float. A single, non-numeric class is a (trivial) classification
        problem."""
        assert infer_task(np.array(["foo"] * 20), ["label"]) == "classification"


class TestLoadDataset:
    def test_requires_sequence(self, tmp_path):
        path = _write(tmp_path, "bad.csv", pd.DataFrame({"seq": ["AC"], "label": [1]}))
        with pytest.raises(ValueError, match="missing required column 'sequence'"):
            load_dataset(path)

    def test_reports_missing_label_column(self, tmp_path):
        path = _write(tmp_path, "nolabel.csv", _frame())
        with pytest.raises(ValueError, match="no label column"):
            load_dataset(path)

    def test_labels_column_is_accepted(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(labels=np.arange(200) * 0.5))
        assert load_dataset(path).task == "regression"

    def test_id_column_is_not_required(self, tmp_path):
        """Most of this repo's own sample data has no `ID` column and nothing
        downstream reads it -- only `sequence` and a label column matter."""
        frame = _frame(label=np.arange(200) * 1.0).drop(columns=["ID"])
        path = _write(tmp_path, "no_id.csv", frame)
        ds = load_dataset(path)
        assert len(ds.train_seqs) > 0

    def test_y_column_is_accepted_as_a_label_fallback(self, tmp_path):
        """`Y` is the label column spelling used by several of this repo's
        own sample datasets (deeploc, solubility, stability, ...)."""
        path = _write(tmp_path, "d.csv", _frame(Y=np.arange(200) * 0.5))
        assert load_dataset(path).task == "regression"

    def test_explicit_label_col_wins_over_y_fallback(self, tmp_path):
        frame = _frame(Y=np.arange(200) * 1.0, real=np.arange(200) * 2.0)
        path = _write(tmp_path, "d.csv", frame)
        ds = load_dataset(path, label_col="real")
        assert np.allclose(sorted(ds.train_y.tolist() + ds.val_y.tolist()), sorted(frame["real"]))

    def test_missing_label_column_error_names_every_fallback(self, tmp_path):
        path = _write(tmp_path, "nolabel.csv", _frame())
        with pytest.raises(ValueError, match="'label'.*'labels'.*'Y'"):
            load_dataset(path)

    @pytest.mark.parametrize("val_name", ["valid", "val"])
    def test_both_validation_split_spellings(self, tmp_path, val_name):
        """HF-derived sets say 'valid'; locally split ones say 'val'."""
        split = ["train"] * 150 + [val_name] * 50
        path = _write(tmp_path, f"{val_name}.csv", _frame(label=np.arange(200) * 1.0, split=split))
        ds = load_dataset(path, sample=None)
        assert (len(ds.train_seqs), len(ds.val_seqs)) == (150, 50)

    def test_split_column_without_validation_level_raises(self, tmp_path):
        path = _write(
            tmp_path,
            "d.csv",
            _frame(label=np.arange(200) * 1.0, split=["train"] * 100 + ["test"] * 100),
        )
        with pytest.raises(ValueError, match="no .* level"):
            load_dataset(path)

    def test_splits_are_generated_when_absent(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0))
        ds = load_dataset(path, sample=None, val_fraction=0.25)
        assert len(ds.val_seqs) == 50 and len(ds.train_seqs) == 150

    def test_subsampling_respects_the_budget(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        ds = load_dataset(path, sample=140)
        assert len(ds.train_seqs) <= 112 and len(ds.val_seqs) <= 28

    def test_sample_above_one_is_an_absolute_count(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        ds = load_dataset(path, sample=100)
        assert len(ds.train_seqs) + len(ds.val_seqs) <= 100

    def test_sample_between_zero_and_one_is_a_fraction_of_the_dataset(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        ds = load_dataset(path, sample=0.1)
        # 10% of the whole (post-cleaning) dataset, not just the train split.
        assert len(ds.train_seqs) + len(ds.val_seqs) <= 100

    def test_sample_non_positive_raises(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        with pytest.raises(ValueError, match="sample must be > 0"):
            load_dataset(path, sample=0)

    def test_stratified_subsampling_keeps_every_class(self, tmp_path):
        """Rare classes must survive the cut, or the probe silently degrades."""
        labels = np.concatenate([np.repeat(np.arange(20), 49), np.arange(20)])
        path = _write(tmp_path, "d.csv", _frame(len(labels), label=labels))
        ds = load_dataset(path, sample=300, task="classification")
        assert len(np.unique(ds.train_y)) == 20

    def test_stratified_subsampling_survives_rounding_overshoot(self, tmp_path):
        """Regression test: rounding each class's share up then clamping the
        *total* back down to `n` used to trim uniformly across the whole
        pool, which can zero out a rare class that the per-class rounding
        had just protected."""
        labels = np.concatenate([np.repeat(0, 998), np.repeat(1, 2)])
        path = _write(tmp_path, "d.csv", _frame(len(labels), label=labels))
        ds = load_dataset(path, sample=10, task="classification")
        assert set(np.unique(ds.train_y) | np.unique(ds.val_y)) == {0, 1}

    def test_string_labels_are_encoded_not_crashed(self, tmp_path):
        """Regression test: `label` of string classes used to crash with
        `invalid literal for int()` -- nothing mapped the strings to
        integers before y.astype(int)."""
        labels = ["alpha", "beta", "gamma"] * 10
        path = _write(tmp_path, "d.csv", _frame(30, label=labels))
        ds = load_dataset(path, sample=None)
        assert ds.task == "classification"
        assert ds.train_y.dtype.kind == "i"
        assert set(ds.train_y.tolist()) == {0, 1, 2}

    def test_ambiguous_integer_column_raises(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(210, label=list(np.repeat(np.arange(7), 30))))
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_constant_string_label_column_loads_as_classification(self, tmp_path):
        """Regression test: this used to reach `infer_task`'s constant-column
        shortcut before the numeric-dtype check, get labeled 'regression',
        and crash in `build()` with `could not convert string to float`."""
        path = _write(tmp_path, "d.csv", _frame(20, label=["foo"] * 20))
        ds = load_dataset(path, sample=None)
        assert ds.task == "classification"

    def test_explicit_task_resolves_the_ambiguous_case(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(210, label=list(np.repeat(np.arange(7), 30))))
        ds = load_dataset(path, sample=None, task="classification")
        assert ds.task == "classification"
        assert len(set(ds.train_y.tolist()) | set(ds.val_y.tolist())) == 7

    def test_multi_label_needs_multiple_columns(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0))
        with pytest.raises(ValueError, match="multi-label"):
            load_dataset(path, task="multi-label")

    def test_multi_label_names_the_actually_non_numeric_column(self, tmp_path):
        """Regression test: the non-numeric check used to test the whole
        stacked label array on every column, so every column -- including
        genuinely numeric ones -- was reported as non-numeric."""
        n = 40
        path = _write(
            tmp_path,
            "d.csv",
            _frame(n, a=np.arange(n) % 2, b=["x", "y"] * (n // 2)),
        )
        with pytest.raises(ValueError, match=r"non-numeric \['b'\]"):
            load_dataset(path, task="multi-label", label_col=["a", "b"])

    def test_train_val_share_one_code_space(self, tmp_path):
        """A class present in only one split still gets a stable code, so
        predictions computed on val line up with the codes trained on."""
        labels = ["a"] * 90 + ["b"] * 90 + ["rare"] * 20
        path = _write(tmp_path, "d.csv", _frame(200, label=labels))
        ds = load_dataset(path, sample=None)
        all_codes = set(ds.train_y.tolist()) | set(ds.val_y.tolist())
        assert all_codes == {0, 1, 2}


class TestAvailableRows:
    """`n_train_available`/`n_val_available` let a confidence report tell
    "raise --sample" apart from "you're already using every row in the file"."""

    def test_recorded_when_sampling_binds(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        ds = load_dataset(path, sample=100, val_fraction=0.2)
        # 1000 rows, 80/20 split -> 800 train / 200 val available pre-cap.
        assert ds.n_train_available == 800
        assert ds.n_val_available == 200
        assert len(ds.train_seqs) + len(ds.val_seqs) == 100

    def test_equals_used_when_sampling_does_not_bind(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(200, label=np.arange(200) * 1.0))
        ds = load_dataset(path, sample=None)
        assert ds.n_train_available == len(ds.train_seqs)
        assert ds.n_val_available == len(ds.val_seqs)

    def test_counted_after_max_seq_len_filtering(self, tmp_path):
        """A dropped-too-long sequence must not count as "available", or the
        remedy would tell the user to raise --sample when there's nothing
        left in the file to sample."""
        rows = _frame(200, label=np.arange(200) * 1.0)
        rows.loc[0, "sequence"] = "A" * 5000
        path = _write(tmp_path, "d.csv", rows)
        ds = load_dataset(path, sample=None, max_seq_len=2000)
        assert ds.n_train_available + ds.n_val_available == 199


class TestMaxSeqLen:
    """A single pathological-length sequence must never survive into a run.

    Attention memory is quadratic in length, so one outlier protein can OOM a
    batch regardless of `max_tokens_per_batch` (`model.py`'s `_batches` only
    bounds the *product* of batch size and max length). Dropping it here,
    before the split and the subsample, is what makes that impossible rather
    than merely unlikely.
    """

    def _frame_with_outlier(self, n=200, outlier_len=5000):
        seqs = ["ACDEFGHIKLMNPQRSTVWY"[: 6 + i % 12] for i in range(n)]
        seqs[0] = "A" * outlier_len
        return pd.DataFrame({"sequence": seqs, "label": np.arange(n) * 1.0})

    def test_default_cap_drops_the_outlier_and_warns(self, tmp_path):
        path = _write(tmp_path, "d.csv", self._frame_with_outlier())
        with pytest.warns(RuntimeWarning, match="dropping 1 sequence"):
            ds = load_dataset(path, sample=None)
        all_seqs = ds.train_seqs + ds.val_seqs
        assert all(len(s) <= 2000 for s in all_seqs)
        assert len(all_seqs) == 199

    def test_cap_applies_before_subsampling_not_after(self, tmp_path):
        """A naive 'subsample then drop' order could still let the outlier
        through if it happened to land in the sample -- applying the cap
        first rules that out regardless of sampling."""
        path = _write(tmp_path, "d.csv", self._frame_with_outlier(n=50, outlier_len=5000))
        with pytest.warns(RuntimeWarning):
            ds = load_dataset(path, sample=2000, max_seq_len=2000)
        assert all(len(s) <= 2000 for s in ds.train_seqs + ds.val_seqs)

    def test_none_disables_the_cap(self, tmp_path):
        path = _write(tmp_path, "d.csv", self._frame_with_outlier(n=50, outlier_len=5000))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds = load_dataset(path, sample=None, max_seq_len=None)
        assert any(len(s) == 5000 for s in ds.train_seqs + ds.val_seqs)

    def test_zero_disables_the_cap(self, tmp_path):
        path = _write(tmp_path, "d.csv", self._frame_with_outlier(n=50, outlier_len=5000))
        ds = load_dataset(path, sample=None, max_seq_len=0)
        assert any(len(s) == 5000 for s in ds.train_seqs + ds.val_seqs)

    def test_cap_that_empties_the_dataset_raises_clearly(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(20, label=np.arange(20) * 1.0))
        with pytest.raises(ValueError, match="max_seq_len"):
            load_dataset(path, max_seq_len=1)


class TestBlankLabelRows:
    """The report's exact reproduction: a stray blank label row -- discarded
    either way -- must not change the outcome for the surviving rows."""

    def _class_codes(self, n_rows_per_class=30):
        return list(np.repeat(np.arange(5), n_rows_per_class))

    def test_stray_blank_row_does_not_flip_ambiguous_to_regression(self, tmp_path):
        clean = _frame(150, label=self._class_codes())
        path = _write(tmp_path, "clean.csv", clean)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

        with_blank = _frame(151, label=self._class_codes() + [None])
        path = _write(tmp_path, "blank.csv", with_blank)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_explicit_task_still_resolves_it_regardless_of_the_blank_row(self, tmp_path):
        with_blank = _frame(151, label=self._class_codes() + [None])
        path = _write(tmp_path, "blank.csv", with_blank)
        ds = load_dataset(path, sample=None, task="classification")
        assert ds.task == "classification"
        assert len(set(ds.train_y.tolist()) | set(ds.val_y.tolist())) == 5

    def test_blank_sequence_row_does_not_affect_inference(self, tmp_path):
        """A NaN in `sequence` (not the label) is unrelated to the label
        dtype and must not change task inference either."""
        n = 150
        frame = _frame(n, label=self._class_codes())
        frame.loc[0, "sequence"] = None
        path = _write(tmp_path, "d.csv", frame)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)


class TestStructuredLabelColumns:
    """A single label column holding list/delimited/per-residue values must
    raise, naming the column -- not silently become multi-class
    classification over an explosion of near-unique 'classes'."""

    @pytest.mark.parametrize(
        "bad_value",
        ["0,1,1,0", "[0, 1]", "0;1;1", "0|1|0", "1 0 1"],
    )
    def test_delimited_numeric_strings_raise(self, tmp_path, bad_value):
        labels = [bad_value] * 5 + ["9,9"] * 5
        path = _write(tmp_path, "d.csv", _frame(10, label=labels))
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_list_values_raise(self, tmp_path):
        frame = _frame(10, label=[[0, 1]] * 10)
        path = _write(tmp_path, "d.csv", frame)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_per_residue_strings_raise(self, tmp_path):
        n = 20
        frame = _frame(n)
        frame["label"] = [s.replace("A", "0").replace("C", "1") for s in frame["sequence"]]
        # every label string is exactly as long as its own sequence
        assert all(len(a) == len(b) for a, b in zip(frame["label"], frame["sequence"], strict=True))
        path = _write(tmp_path, "d.csv", frame)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_plain_string_classes_are_unaffected(self, tmp_path):
        """Negative control: ordinary (even multi-word) string classes must
        not be mistaken for a structured label."""
        labels = ["alpha", "DNA binding", "gamma"] * 10
        path = _write(tmp_path, "d.csv", _frame(30, label=labels))
        ds = load_dataset(path, sample=None)
        assert ds.task == "classification"

    def test_genuine_multi_label_still_loads(self, tmp_path):
        """Negative control: real one-numeric-column-per-label multi-label
        data must keep working."""
        n = 30
        frame = _frame(n, a=np.zeros(n), b=np.ones(n))
        path = _write(tmp_path, "d.csv", frame)
        ds = load_dataset(path, sample=None, label_col=["a", "b"])
        assert ds.task == "multi-label"
