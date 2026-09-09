"""Layer selection: the plateau rule, gain reporting, probes and metrics."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from plmsommelier.data import Dataset
from plmsommelier.select import (
    Result,
    _choose,
    _paired_margin,
    _plateau_agreement,
    _seed_spread,
    fit_predict,
    score,
    select_layer,
)


class TestPlateauRule:
    # A real fluorescence-task curve, where layers 2 and 4 are a coin flip.
    CURVE = {0: 0.4774, 1: 0.5057, 2: 0.5999, 3: 0.5845, 4: 0.6031, 5: 0.5227, 6: 0.5802}

    def test_zero_tolerance_is_plain_argmax(self):
        assert _choose(self.CURVE, 0.0) == (4, [4])

    def test_prefers_the_shallowest_indistinguishable_layer(self):
        """Same score, half the blocks -- and a far more stable choice."""
        chosen, plateau = _choose(self.CURVE, 0.02)
        assert chosen == 2
        assert plateau == [2, 4]

    def test_wider_tolerance_widens_the_plateau(self):
        assert _choose(self.CURVE, 0.05)[1] == [2, 3, 4, 6]

    def test_nan_layers_are_skipped(self):
        curve = {0: float("nan"), 1: 0.4, 2: 0.8}
        assert _choose(curve, 0.0)[0] == 2

    def test_all_nan_raises(self):
        with pytest.raises(RuntimeError, match="degenerate"):
            _choose({0: float("nan")}, 0.0)


class TestGainReporting:
    def _result(self, **kw):
        base = dict(
            model="m",
            dataset="d",
            task="regression",
            probe="knn",
            best_layer=2,
            best_score=0.60,
            last_layer=6,
            last_layer_score=0.50,
            curve={2: 0.60, 6: 0.50},
            plateau=[2],
            n_train=100,
            n_val=50,
        )
        base.update(kw)
        return Result(**base)

    def test_gain_over_last_layer(self):
        assert self._result().gain_over_last == pytest.approx(0.2)

    def test_depth_fraction(self):
        assert self._result().depth_fraction == pytest.approx(2 / 6)

    def test_zero_last_layer_score_is_not_a_division_error(self):
        assert np.isnan(self._result(last_layer_score=0.0).gain_over_last)

    def test_negative_last_layer_score_does_not_flip_sign(self):
        # A signed metric (Pearson/MCC) against a negative baseline has no
        # meaningful "percent improvement" -- best=0.5 over last=-0.5 must
        # not read as -200%.
        r = self._result(best_score=0.5, last_layer_score=-0.5)
        assert np.isnan(r.gain_over_last)
        assert "n/a" in r.summary()

    def test_summary_mentions_the_headline_numbers(self):
        text = self._result().summary()
        assert "best layer" in text and "gain over last" in text and "+20.0%" in text

    def test_curve_plot_renders_bars_nan_and_tags(self):
        r = self._result(
            best_layer=2,
            last_layer=3,
            curve={1: 0.3, 2: 0.5, 3: float("nan")},
        )
        text = r.curve_plot()
        lines = text.split("\n")
        assert lines[0] == ""
        assert "layer performance (pearson):" in lines[1]
        assert any("<- best" in ln for ln in lines)
        assert any(ln.strip().endswith("nan") for ln in lines)
        assert not any("(last)" in ln for ln in lines)  # layer 3 is nan, not scored

    def test_curve_plot_empty_when_all_nan(self):
        r = self._result(curve={1: float("nan"), 2: float("nan")})
        assert r.curve_plot() == ""

    def test_result_serialises(self):
        d = self._result().to_dict()
        assert "gain_over_last" in d and "curve" in d
        json.loads(json.dumps(d, default=str))


def _build_result(**kw) -> Result:
    base = dict(
        model="m",
        dataset="d",
        task="regression",
        probe="knn",
        best_layer=2,
        best_score=0.60,
        last_layer=6,
        last_layer_score=0.50,
        curve={2: 0.60, 6: 0.50},
        plateau=[2],
        n_train=100,
        n_val=50,
    )
    base.update(kw)
    return Result(**base)


class TestSeedStatistics:
    """The pure functions behind the confidence signals."""

    def test_plateau_agreement_counts_adjacent_tied_layers(self):
        """The real bench case: seed_agreement (exact-match) is 0.0 while
        every seed actually landed inside the plateau."""
        plateau = list(range(13, 30))  # esm2_650m/homology's actual plateau
        seed_layers = [17, 16, 14]
        assert _plateau_agreement(seed_layers, plateau) == pytest.approx(1.0)

    def test_plateau_agreement_with_a_seed_outside(self):
        assert _plateau_agreement([2, 2, 5], [1, 2, 3]) == pytest.approx(2 / 3)

    def test_plateau_agreement_nan_without_seeds(self):
        assert math.isnan(_plateau_agreement([], [2]))

    def test_seed_spread_is_stable_in_the_number_of_seeds(self):
        """(max - min) would shrink as more seeds are drawn from the same
        distribution -- mean absolute deviation doesn't, so raising
        --n-seeds sharpens the measurement instead of the apparent stability."""
        spread_3 = _seed_spread([0, 4, 8], best=4, n_layers=10)
        spread_9 = _seed_spread([0, 4, 8] * 3, best=4, n_layers=10)
        assert spread_3 == pytest.approx(spread_9)

    def test_paired_margin_cancels_a_shared_seed_offset(self):
        base = [
            {0: 0.1, 1: 0.9, 2: 0.2},
            {0: 0.15, 1: 0.85, 2: 0.25},
            {0: 0.05, 1: 0.95, 2: 0.15},
        ]
        shifted = [{layer: v + 0.5 for layer, v in c.items()} for c in base]
        m1 = _paired_margin(base, plateau=[1])
        m2 = _paired_margin(shifted, plateau=[1])
        assert m1 == pytest.approx(m2)

    def test_paired_margin_is_inf_when_every_seed_agrees_exactly(self):
        curves = [{0: 0.1, 1: 0.9}] * 3
        assert _paired_margin(curves, plateau=[1]) == float("inf")

    def test_paired_margin_nan_when_plateau_spans_every_layer(self):
        curves = [{0: 0.1, 1: 0.9}] * 3
        assert math.isnan(_paired_margin(curves, plateau=[0, 1]))

    def test_paired_margin_abstains_with_fewer_than_two_usable_seeds(self):
        curves = [{0: 0.1, 1: 0.9}]
        assert math.isnan(_paired_margin(curves, plateau=[1]))


class TestConfidenceVerdict:
    """The weakest-link verdict ladder, driven by hand-built Results."""

    def test_no_seeds_is_unmeasured_not_low(self):
        r = _build_result(seed_layers=[], seed_curves=[])
        assert r.confidence == "unmeasured"

    def test_single_layer_model_is_high(self):
        r = _build_result(curve={0: 0.5}, plateau=[0], best_layer=0, last_layer=0)
        assert r.confidence == "high"

    def test_wide_plateau_does_not_get_promoted_by_agreement_alone(self):
        """esm2_650m/homology shape: a 17-of-34-layer plateau that every seed
        pick falls inside. 100% plateau agreement here means almost nothing
        -- agreement must abstain, not vote "high"."""
        plateau = list(range(13, 30))
        curve = {layer: (0.9 if layer in plateau else 0.5) for layer in range(34)}
        r = _build_result(
            curve=curve,
            plateau=plateau,
            best_layer=13,
            last_layer=33,
            seed_layers=[17, 16, 14],
            seed_curves=[dict(curve)] * 3,
        )
        assert r.confidence != "high"

    def test_scattered_picks_outside_a_narrow_plateau_are_low(self):
        """progen2_medium/meltome shape: every seed disagreed with a narrow
        plateau near the shallow end of the network."""
        r = _build_result(
            curve={0: 0.9, 1: 0.88, 26: 0.5},
            plateau=[0, 1],
            best_layer=0,
            last_layer=26,
            seed_layers=[26, 26, 26],
            seed_curves=[{0: 0.3, 1: 0.32, 26: 0.9}] * 3,
        )
        assert r.confidence == "low"

    def test_weakest_link_not_average(self):
        """A strong margin cannot rescue scattered seed picks -- min, not mean."""
        r = _build_result(
            curve={0: 0.1, 1: 0.9, 2: 0.15},
            plateau=[1],
            best_layer=1,
            last_layer=10,
            seed_layers=[1, 9, 0],  # unanimous on the huge margin, wildly scattered
            seed_curves=[
                {0: 0.1, 1: 0.95, 2: 0.1},
                {0: 0.1, 1: 0.85, 2: 0.15},
                {0: 0.1, 1: 0.9, 2: 0.2},
            ],
        )
        assert r.confidence == "low"

    def test_unanimous_narrow_plateau_is_high(self):
        r = _build_result(
            curve={0: 0.5, 1: 0.51, 2: 0.9, 3: 0.52, 4: 0.5},
            plateau=[2],
            best_layer=2,
            last_layer=4,
            seed_layers=[2, 2, 2],
            seed_curves=[
                {0: 0.5, 1: 0.51, 2: 0.9, 3: 0.52, 4: 0.5},
                {0: 0.49, 1: 0.5, 2: 0.89, 3: 0.51, 4: 0.49},
                {0: 0.51, 1: 0.52, 2: 0.91, 3: 0.53, 4: 0.51},
            ],
        )
        assert r.confidence == "high"

    def test_flat_curve_is_low_not_unmeasured(self):
        curve = {layer: 0.5 for layer in range(5)}
        r = _build_result(
            curve=curve,
            plateau=list(range(5)),
            best_layer=0,
            last_layer=4,
            seed_layers=[0, 1, 2],
            seed_curves=[curve] * 3,
        )
        assert r.confidence == "low"


class TestRemedies:
    def test_no_remedies_when_confidence_is_high(self):
        r = _build_result(curve={0: 0.5}, plateau=[0], best_layer=0, last_layer=0)
        assert r.remedies == []

    def test_exhausted_file_says_sample_cannot_help(self):
        """Weakness #4's regression test: when every available row is
        already in use, the remedy must never say "raise --sample"."""
        r = _build_result(
            seed_layers=[6, 6, 6],
            seed_curves=[{2: 0.5, 6: 0.9}] * 3,
            n_train_available=100,
            n_val_available=50,
        )
        assert any("cannot help" in m for m in r.remedies)
        assert not any("raise --sample" in m for m in r.remedies)

    def test_binding_sample_recommends_raising_it(self):
        r = _build_result(
            seed_layers=[6, 6, 6],
            seed_curves=[{2: 0.5, 6: 0.9}] * 3,
            n_train_available=1000,
            n_val_available=500,
        )
        assert any("raise --sample" in m for m in r.remedies)

    def test_unknown_availability_makes_no_sampling_claim(self):
        r = _build_result(seed_layers=[6, 6, 6], seed_curves=[{2: 0.5, 6: 0.9}] * 3)
        assert not any("available rows" in m for m in r.remedies)

    def test_flat_curve_advice_accepts_the_plateau(self):
        curve = {layer: 0.5 for layer in range(5)}
        r = _build_result(
            curve=curve,
            plateau=list(range(5)),
            best_layer=0,
            last_layer=4,
            seed_layers=[0, 1, 2],
            seed_curves=[curve] * 3,
        )
        hint = next(m for m in r.remedies if "genuinely flat" in m)
        assert "correct answer" in hint

    def test_knn_suggests_a_linear_cross_check(self):
        r = _build_result(probe="knn", seed_layers=[6, 6, 6], seed_curves=[{2: 0.5, 6: 0.9}] * 3)
        assert any("--probe lr" in m for m in r.remedies)

    def test_lr_does_not_suggest_switching_probes(self):
        r = _build_result(probe="lr", seed_layers=[6, 6, 6], seed_curves=[{2: 0.5, 6: 0.9}] * 3)
        assert not any("--probe lr" in m for m in r.remedies)


class TestCurvePlotErrorBars:
    def test_whiskers_render_when_sigma_is_available(self):
        r = _build_result(
            curve={0: 0.5, 2: 0.9},
            plateau=[2],
            best_layer=2,
            last_layer=2,
            seed_layers=[2, 2],
            seed_curves=[{0: 0.4, 2: 0.85}, {0: 0.6, 2: 0.95}],
        )
        text = r.curve_plot()
        assert "+/-" in text
        assert "1 sd across 2 seeds" in text

    def test_falls_back_to_the_plain_bar_without_seeds(self):
        curve = {1: 0.3, 2: 0.5}
        no_seeds = _build_result(curve=curve, plateau=[2], best_layer=2, last_layer=2)
        # a single seed can't produce a std (ddof=1 needs >= 2 draws), so
        # sigma is all-NaN here too -- rendering must fall back identically.
        one_seed = _build_result(
            curve=curve,
            plateau=[2],
            best_layer=2,
            last_layer=2,
            seed_layers=[2],
            seed_curves=[{1: 0.28, 2: 0.48}],
        )
        assert no_seeds.curve_plot() == one_seed.curve_plot()

    def test_nan_layers_still_render_as_nan(self):
        r = _build_result(
            curve={1: 0.3, 2: 0.5, 3: float("nan")},
            plateau=[2],
            best_layer=2,
            last_layer=3,
            seed_layers=[2, 2],
            seed_curves=[{1: 0.28, 2: 0.48}, {1: 0.32, 2: 0.52}],
        )
        assert any(ln.strip().endswith("nan") for ln in r.curve_plot().split("\n"))


class TestScore:
    def test_perfect_regression_scores_one(self):
        y = np.linspace(0, 1, 50)
        assert score(y, y, "regression") == pytest.approx(1.0)

    def test_constant_prediction_is_nan_not_a_crash(self):
        assert np.isnan(score(np.ones(10), np.arange(10.0), "regression"))

    def test_perfect_classification_scores_one(self):
        y = np.tile([0, 1], 25)
        assert score(y, y, "classification") == pytest.approx(1.0)

    def test_degenerate_multi_label_column_contributes_zero(self):
        y = np.zeros((50, 2), dtype=int)  # column 0 all-negative
        y[:, 1] = np.tile([0, 1], 25)
        assert score(y, y, "multi-label") == pytest.approx(0.5)


class TestFitPredict:
    def test_multi_label_returns_one_column_per_task(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 5))
        y = (x[:, :3] > 0).astype(int)
        pred = fit_predict(x[:60], y[:60], x[60:], y[60:], task="multi-label", probe="knn")
        assert pred.shape == (40, 3)

    def test_unknown_probe_raises(self):
        with pytest.raises(ValueError, match="probe"):
            fit_predict(
                np.zeros((5, 2)),
                np.zeros(5),
                np.zeros((5, 2)),
                np.zeros(5),
                task="regression",
                probe="xyz",
            )

    def test_empty_train_set_raises_a_clear_error(self):
        """Regression test: an empty `train_y` used to hit `train_y[0]` in the
        degenerate-split shortcut and raise a bare IndexError."""
        with pytest.raises(ValueError, match="empty"):
            fit_predict(
                np.zeros((0, 2)),
                np.zeros(0),
                np.zeros((5, 2)),
                np.zeros(5),
                task="regression",
            )


class TestSelectLayer:
    """End-to-end selection on synthetic embeddings with a known best layer."""

    def _fixture(self, n_states=5, best=1, n=200, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.normal(size=n)
        values = np.zeros((n_states, n, dim), dtype=np.float16)
        for layer in range(n_states):
            # signal peaks at `best` and decays away from it
            strength = 1.0 / (1.0 + 2.0 * abs(layer - best))
            signal = np.outer(y, rng.normal(size=dim)) * strength
            values[layer] = (signal + rng.normal(scale=0.3, size=(n, dim))).astype(np.float16)
        cut = n // 2
        ds = Dataset(
            name="synthetic",
            task="regression",
            train_seqs=["A"] * cut,
            train_y=y[:cut],
            val_seqs=["A"] * (n - cut),
            val_y=y[cut:],
        )
        return ds, values[:, :cut], values[:, cut:]

    def test_finds_the_planted_layer(self):
        ds, train, val = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=0, tolerance=0.0)
        assert res.best_layer == 1

    def test_curve_covers_every_layer(self):
        ds, train, val = self._fixture()
        res = select_layer(ds, train, val, n_seeds=0)
        assert sorted(res.curve) == [0, 1, 2, 3, 4]

    def test_notes_when_the_last_layer_wins(self):
        ds, train, val = self._fixture(best=4)
        res = select_layer(ds, train, val, n_seeds=0, tolerance=0.0)
        assert res.best_layer == 4
        assert any("last layer won" in n for n in res.notes)

    def test_seed_agreement_is_reported(self):
        ds, train, val = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        assert len(res.seed_layers) == 3
        assert 0.0 <= res.seed_agreement <= 1.0

    def test_seed_curves_cover_every_layer_for_every_seed(self):
        ds, train, val = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        assert len(res.seed_curves) == 3
        for curve in res.seed_curves:
            assert sorted(curve) == sorted(res.curve)

    def test_seed_curves_kept_at_least_as_often_as_seed_layers(self):
        ds, train, val = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        assert len(res.seed_curves) >= len(res.seed_layers)

    def test_val_rows_are_resampled_across_seeds(self, monkeypatch):
        """The default fixture has n_val=100 (>= the 40-row floor), so val
        rows must actually vary seed to seed, not just train rows."""
        import plmsommelier.select as select_mod

        ds, train, val = self._fixture(best=1)
        seen_val_lengths = []
        real_fit_predict = select_mod.fit_predict

        def _spy(train_x, train_y, val_x, val_y, **kw):
            seen_val_lengths.append(len(val_x))
            return real_fit_predict(train_x, train_y, val_x, val_y, **kw)

        monkeypatch.setattr(select_mod, "fit_predict", _spy)
        select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        # the main run uses the full val set (100); resampled seeds use 80%.
        assert 100 in seen_val_lengths
        assert 80 in seen_val_lengths

    def test_val_is_not_resampled_below_the_floor(self, monkeypatch):
        import plmsommelier.select as select_mod

        ds, train, val = self._fixture(best=1, n=60)  # n_val = 30, below the 40-row floor
        seen_val_lengths = set()
        real_fit_predict = select_mod.fit_predict

        def _spy(train_x, train_y, val_x, val_y, **kw):
            seen_val_lengths.add(len(val_x))
            return real_fit_predict(train_x, train_y, val_x, val_y, **kw)

        monkeypatch.setattr(select_mod, "fit_predict", _spy)
        select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        assert seen_val_lengths == {30}

    def test_result_carries_availability_from_the_dataset(self):
        ds, train, val = self._fixture(best=1)
        ds.n_train_available, ds.n_val_available = 5000, 2000
        res = select_layer(ds, train, val, n_seeds=0, tolerance=0.0)
        assert res.n_train_available == 5000
        assert res.n_val_available == 2000

    def test_result_serialises_with_seed_curves(self):
        ds, train, val = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        d = res.to_dict()
        json.loads(json.dumps(d, default=str))
        assert "confidence" in d
        assert len(d["seed_curves"]) == 3


class TestFeatureScaling:
    """Unnormalized Euclidean KNN lets a few high-variance ("rogue") feature
    dimensions dominate the distance calculation, independent of how
    informative the layer actually is (BUG_REPORT.md MEDIUM #1)."""

    def _rogue_fixture(self, n=240, seed=0):
        """One informative small-scale feature plus one huge-variance,
        uninformative "rogue" feature. Without per-feature scaling, KNN's
        Euclidean distance is dominated by the rogue feature and the probe
        can't recover the real signal at all."""
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        signal = np.where(y == 1, 1.0, -1.0) + rng.normal(scale=0.3, size=n)
        rogue = rng.normal(scale=1000.0, size=n)  # carries no information about y
        x = np.stack([signal, rogue], axis=1).astype(np.float32)
        cut = n // 2
        return x[:cut], y[:cut], x[cut:], y[cut:]

    def test_knn_recovers_signal_drowned_out_by_an_unscaled_rogue_dimension(self):
        train_x, train_y, val_x, val_y = self._rogue_fixture()
        pred = fit_predict(train_x, train_y, val_x, val_y, task="classification", probe="knn")
        assert score(pred, val_y, "classification") > 0.5

    def test_lr_recovers_signal_drowned_out_by_an_unscaled_rogue_dimension(self):
        train_x, train_y, val_x, val_y = self._rogue_fixture()
        pred = fit_predict(train_x, train_y, val_x, val_y, task="classification", probe="lr")
        assert score(pred, val_y, "classification") > 0.5

    def test_constant_feature_does_not_produce_nan(self):
        rng = np.random.default_rng(1)
        n = 60
        y = rng.normal(size=n)
        x = np.stack([y, np.zeros(n)], axis=1).astype(np.float32)
        cut = n // 2
        pred = fit_predict(x[:cut], y[:cut], x[cut:], y[cut:], task="regression", probe="knn")
        assert not np.isnan(pred).any()

    def test_select_layer_is_not_misled_by_a_rogue_layer(self):
        """End-to-end: a layer with real (if small-scale) signal must still
        win over a layer whose only distinguishing feature is a huge-variance
        dimension carrying no information about the label."""
        rng = np.random.default_rng(2)
        n = 240
        y = rng.normal(size=n)
        dim = 4

        real_signal = np.outer(y, rng.normal(size=dim))
        good_layer = (real_signal + rng.normal(scale=0.3, size=(n, dim))).astype(np.float32)

        rogue_layer = rng.normal(scale=0.3, size=(n, dim))
        rogue_layer[:, 0] += rng.normal(scale=1000.0, size=n)  # uninformative rogue dimension
        rogue_layer = rogue_layer.astype(np.float32)

        values = np.stack([rogue_layer, good_layer]).astype(np.float16)
        cut = n // 2
        ds = Dataset(
            name="synthetic",
            task="regression",
            train_seqs=["A"] * cut,
            train_y=y[:cut],
            val_seqs=["A"] * (n - cut),
            val_y=y[cut:],
        )
        res = select_layer(ds, values[:, :cut], values[:, cut:], n_seeds=0, tolerance=0.0)
        assert res.best_layer == 1
