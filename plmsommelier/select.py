"""The layer search: fit a probe per layer, pick the best one."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plmsommelier.data import Dataset, Task

__all__ = ["Result", "select_layer", "fit_predict"]

# Confidence-verdict thresholds. Heuristic, not calibrated probabilities --
# see `Result.confidence`'s docstring for what each signal actually measures
# and why these particular cutoffs.
_HIGH_AGREEMENT = 0.8
_MOD_AGREEMENT = 0.5
_HIGH_MARGIN = 2.0
_MOD_MARGIN = 1.0
_HIGH_SPREAD = 0.05
_MOD_SPREAD = 0.15
# Agreement only means something when the plateau is a small target to hit;
# past this fraction of the (finite) layers, landing "inside" it is close to
# guaranteed and the signal abstains rather than voting "high".
_WIDE_PLATEAU_FRACTION = 0.5

_VOTE_RANK = {"low": 0, "moderate": 1, "high": 2}

# Result properties that are plain derived values -- Result.to_dict() copies
# each of these in by name. Fields needing a transform (the int-keyed dicts)
# stay written out explicitly in to_dict().
_DERIVED_FIELDS = (
    "metric",
    "gain_over_last",
    "depth_fraction",
    "plateau_agreement",
    "seed_spread",
    "peak_margin",
    "plateau_is_contiguous",
    "sampling_is_binding",
    "confidence",
    "remedies",
)


def _layer(embeddings: np.ndarray, index: int) -> np.ndarray:
    """``(N, D)`` float32 view of one layer, ready for a probe."""
    return embeddings[index].astype(np.float32, copy=False)


def _multioutput_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-column binary MCC. Degenerate columns contribute 0.0, not NaN."""
    y_true, y_pred = y_true.astype(int), y_pred.astype(int)
    scores = []
    for col in range(y_true.shape[1]):
        t, p = y_true[:, col], y_pred[:, col]
        if len(np.unique(t)) < 2 and len(np.unique(p)) < 2:
            scores.append(0.0)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            score = matthews_corrcoef(t, p)
        scores.append(0.0 if np.isnan(score) else float(score))
    return float(np.mean(scores)) if scores else 0.0


def score(pred: np.ndarray, y: np.ndarray, task: Task) -> float:
    """Pearson's r for regression, MCC for classification and multi-label."""
    pred, y = np.asarray(pred), np.asarray(y)
    if task == "regression":
        if np.std(pred) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(pred, y)[0, 1])
    if task == "multi-label":
        return _multioutput_mcc(y, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = matthews_corrcoef(y.astype(int), pred.astype(int))
    return 0.0 if np.isnan(mcc) else float(mcc)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    task: Task,
    probe: str = "knn",
    k: int = 10,
) -> np.ndarray:
    """Fit on train, predict on val."""
    if probe not in ("knn", "lr"):
        raise ValueError(f"probe must be 'knn' or 'lr', got {probe!r}")
    if len(train_y) == 0:
        raise ValueError("train_y is empty -- nothing to fit")

    if task == "multi-label":
        cols = [
            fit_predict(
                train_x, train_y[:, c], val_x, val_y[:, c], task="classification", probe=probe, k=k
            )
            for c in range(train_y.shape[1])
        ]
        return np.stack(cols, axis=1)

    if task == "classification" and len(np.unique(train_y)) < 2:
        # Degenerate split: a constant predictor is the honest answer.
        return np.full(len(val_x), float(train_y[0]))

    if probe == "knn":
        cls = KNeighborsRegressor if task == "regression" else KNeighborsClassifier
        estimator = cls(n_neighbors=max(1, min(k, len(train_x))), n_jobs=-1)
    elif task == "regression":
        estimator = LinearRegression()
    else:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")

    # Standardize per feature, fit on train only, so the cross-layer score
    # comparison is fair: without this, a few high-variance ("rogue") feature
    # dimensions in one layer's activations can dominate KNN's Euclidean
    # distance regardless of how informative that layer actually is.
    # StandardScaler maps a zero-variance (degenerate) feature to scale 1, so
    # it never introduces a NaN/inf on its own.
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(val_x)).ravel()


def _layer_sigma(seed_curves: list[dict[int, float]], layers) -> dict[int, float]:
    """Std of each layer's score across seeds. NaN with fewer than 2 finite draws."""
    out: dict[int, float] = {}
    for layer in layers:
        vals = [c[layer] for c in seed_curves if layer in c and math.isfinite(c[layer])]
        out[layer] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")
    return out


def _plateau_agreement(seed_layers: list[int], plateau: list[int]) -> float:
    """Fraction of seed picks landing anywhere in the main run's plateau.

    Unlike exact-match ``seed_agreement``, this respects the plateau rule: a
    seed that picks a neighbouring, statistically tied layer counts as
    agreement, not disagreement.
    """
    if not seed_layers:
        return float("nan")
    plateau_set = set(plateau)
    return sum(layer in plateau_set for layer in seed_layers) / len(seed_layers)


def _seed_spread(seed_layers: list[int], best: int, n_layers: int) -> float:
    """Mean absolute deviation of seed picks from ``best``, as a fraction of depth.

    Mean absolute deviation rather than ``(max - min)``: the expected range of
    a sample grows with its size, so a range-based spread would shrink toward
    "more stable" purely from raising ``--n-seeds`` -- exactly backwards, since
    more seeds should sharpen the *measurement*, not the apparent stability.
    """
    if not seed_layers:
        return float("nan")
    denom = max(1, n_layers)
    return float(np.mean([abs(layer - best) for layer in seed_layers])) / denom


def _paired_margin(seed_curves: list[dict[int, float]], plateau: list[int]) -> float:
    """How many (paired) standard deviations separate the plateau from the rest.

    For each seed, ``d = max(score inside plateau) - max(score outside
    plateau)``; the ratio is ``mean(d) / std(d)``. Pairing within each seed
    cancels a shared per-seed offset (an easy resample lifts every layer's
    score at once) that would otherwise pollute a naive per-layer-sigma
    ratio. This is a resampling *stability* statistic, not a hypothesis
    test -- seeds are overlapping subsamples of one dataset, not independent
    draws, so no p-value is implied and the true variance is understated,
    not overstated.

    NaN when there's no layer outside the plateau to compare against, or
    fewer than two seeds have both a plateau and a non-plateau score.
    +/-inf when every seed agrees exactly (zero variance) -- not floored to
    a large finite number, since that would fabricate precision. Zero when
    the seeds also agree the margin itself is exactly zero.
    """
    plateau_set = set(plateau)
    diffs = []
    for c in seed_curves:
        inside = [s for layer, s in c.items() if layer in plateau_set and math.isfinite(s)]
        outside = [s for layer, s in c.items() if layer not in plateau_set and math.isfinite(s)]
        if inside and outside:
            diffs.append(max(inside) - max(outside))
    if len(diffs) < 2:
        return float("nan")
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    # Floating-point cancellation leaves a spurious ~1e-16 "variance" even
    # when every seed's diff is bit-identical -- treat anything below this
    # floor as exactly zero rather than reporting a fabricated ratio in the
    # quadrillions.
    if std_d < 1e-9 * max(abs(mean_d), 1e-9):
        if mean_d == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean_d)
    return mean_d / std_d


@dataclass(slots=True)
class Result:
    """What the tool concluded, and how much to trust it."""

    model: str
    dataset: str
    task: Task
    probe: str

    best_layer: int
    best_score: float
    last_layer: int
    last_layer_score: float

    curve: dict[int, float]
    plateau: list[int]
    n_train: int
    n_val: int

    seed_layers: list[int] = field(default_factory=list)
    seed_agreement: float = float("nan")
    notes: list[str] = field(default_factory=list)

    tolerance: float = 0.02
    # one layer->score map per seed resample, in the same order as
    # `seed_layers` was appended -- except a seed whose resampled curve was
    # entirely NaN contributes a curve here but no pick to `seed_layers`, so
    # `len(seed_curves) >= len(seed_layers)` always holds.
    seed_curves: list[dict[int, float]] = field(default_factory=list)
    # rows available before `--sample` subsampling; None when unknown (every
    # hand-built Result in the test suite leaves this unset).
    n_train_available: int | None = None
    n_val_available: int | None = None

    @property
    def metric(self) -> str:
        return "pearson" if self.task == "regression" else "mcc"

    @property
    def gain_over_last(self) -> float:
        """Relative improvement of the chosen layer over the conventional last one.

        NaN when ``last_layer_score`` is non-positive: both metrics here
        (Pearson's r, MCC) are signed, so a ratio against a negative or
        near-zero baseline doesn't mean "improvement" -- it can even flip
        sign against a genuine improvement (e.g. best=0.5, last=-0.5 would
        read as -200%). There's no meaningful percentage to report there.
        """
        last = self.last_layer_score
        if not math.isfinite(last) or last < 1e-12:
            return float("nan")
        return self.best_score / last - 1.0

    @property
    def depth_fraction(self) -> float:
        return self.best_layer / self.last_layer if self.last_layer else 0.0

    @property
    def _n_finite_layers(self) -> int:
        return sum(1 for s in self.curve.values() if math.isfinite(s))

    @property
    def layer_sigma(self) -> dict[int, float]:
        """Std of each layer's score across seed resamples (NaN below 2 draws)."""
        return _layer_sigma(self.seed_curves, self.curve.keys())

    @property
    def plateau_agreement(self) -> float:
        """Fraction of seed picks landing anywhere in the plateau (not just on `best`)."""
        return _plateau_agreement(self.seed_layers, self.plateau)

    @property
    def seed_spread(self) -> float:
        """Mean absolute deviation of seed picks from `best`, as a fraction of depth."""
        return _seed_spread(self.seed_layers, self.best_layer, self.last_layer)

    @property
    def peak_margin(self) -> float:
        """Paired separation ratio: how many (paired) std devs separate the
        plateau from the best layer outside it, across seed resamples. See
        `_paired_margin` for what this is and is not."""
        return _paired_margin(self.seed_curves, self.plateau)

    @property
    def plateau_is_contiguous(self) -> bool:
        """Whether the plateau is one unbroken run of layers around the peak.

        A non-contiguous plateau means the tolerance band bridged two
        genuinely different regimes of the curve -- `_choose`'s "take the
        shallowest tied layer" can then hand back a layer from the wrong
        regime entirely.
        """
        if len(self.plateau) <= 1:
            return True
        ordered = sorted(self.plateau)
        # deliberately mismatched lengths (pairwise adjacency check) -- not a
        # strict zip.
        return all(b - a == 1 for a, b in zip(ordered, ordered[1:]))  # noqa: B905

    @property
    def sampling_is_binding(self) -> bool | None:
        """Whether more data is available than `--sample` used.

        None when availability wasn't recorded (e.g. a hand-built `Result`).
        """
        if self.n_train_available is None or self.n_val_available is None:
            return None
        return self.n_train_available > self.n_train or self.n_val_available > self.n_val

    @property
    def confidence(self) -> str:
        """A weakest-link verdict over three independent stability signals.

        "unmeasured" when the stability check wasn't run (`--n-seeds 0`) or
        produced no usable seed picks -- this is distinct from "low": low
        means "we checked, and it isn't stable", unmeasured means "we didn't
        check". A model with only one layer to choose from is always "high"
        -- there is nothing to be unstable about.

        Otherwise this is the minimum of up to three votes (agreement, peak
        margin, seed spread), each independently high/moderate/low, with a
        signal abstaining (contributing no vote) when it isn't informative
        for this curve -- e.g. agreement abstains when the plateau is wide
        enough that landing inside it is nearly guaranteed. If every signal
        abstains, that itself means nothing separates the candidates: "low".
        """
        if self._n_finite_layers <= 1:
            return "high"
        if not self.seed_layers:
            return "unmeasured"

        votes = []

        n_finite = self._n_finite_layers
        wide_plateau = n_finite > 0 and len(self.plateau) > _WIDE_PLATEAU_FRACTION * n_finite
        agreement = self.plateau_agreement
        if not wide_plateau and math.isfinite(agreement):
            if agreement >= _HIGH_AGREEMENT:
                votes.append("high")
            elif agreement >= _MOD_AGREEMENT:
                votes.append("moderate")
            else:
                votes.append("low")

        margin = self.peak_margin
        if not math.isnan(margin):
            m = margin
            if m >= _HIGH_MARGIN:
                votes.append("high")
            elif m >= _MOD_MARGIN:
                votes.append("moderate")
            else:
                votes.append("low")

        spread = self.seed_spread
        if math.isfinite(spread):
            if spread <= _HIGH_SPREAD:
                votes.append("high")
            elif spread <= _MOD_SPREAD:
                votes.append("moderate")
            else:
                votes.append("low")

        if not votes:
            return "low"
        return min(votes, key=_VOTE_RANK.get)

    @property
    def remedies(self) -> list[str]:
        """Concrete, actionable levers for raising confidence in this run.

        Empty when confidence is already "high". Only advice that names
        something the user can actually do differently -- diagnostic-only
        signals (e.g. how many seeds ran) belong in the JSON output, not here.
        """
        if self.confidence == "high":
            return []

        out: list[str] = []

        n_finite = self._n_finite_layers
        wide_plateau = n_finite > 0 and len(self.plateau) > _WIDE_PLATEAU_FRACTION * n_finite
        if wide_plateau and n_finite > 1:
            out.append(
                f"the curve is genuinely flat: layers {self.plateau} score the same within "
                f"tolerance. Low confidence is the correct answer here, not a problem to fix "
                f"-- layer {self.best_layer} is already the shallowest (smallest, fastest) "
                "layer in that tied region."
            )

        binding = self.sampling_is_binding
        if binding is True:
            out.append(
                f"using {self.n_train}+{self.n_val} of "
                f"{self.n_train_available}+{self.n_val_available} available rows -- raise "
                "--sample. This is the lever that actually reduces variance; more seeds only "
                "measure it more precisely."
            )
        elif binding is False:
            out.append(
                f"already using every available row ({self.n_train_available}+"
                f"{self.n_val_available}) -- --sample cannot help here. More confidence needs "
                "more rows in the CSV, a lower-variance probe, or accepting the plateau."
            )

        if self.probe == "knn":
            out.append(
                "cross-check with --probe lr: a linear probe has much lower variance at "
                "typical sample sizes. Two probes agreeing on a layer is stronger evidence "
                "than either one's confidence number alone."
            )

        return out

    def summary(self) -> str:
        gain = self.gain_over_last
        gain_txt = "n/a" if math.isnan(gain) else f"{gain:+.1%}"
        lines = [
            f"model    {self.model}",
            f"dataset  {self.dataset}  ({self.task}, {self.metric}, {self.probe} probe)",
            f"data     {self.n_train} train / {self.n_val} val",
            "",
            f"best layer      {self.best_layer} of {self.last_layer} "
            f"({self.depth_fraction:.0%} depth)   {self.metric} = {self.best_score:.4f}",
            f"last layer      {self.last_layer}"
            f"{' ' * 15}{self.metric} = {self.last_layer_score:.4f}",
            f"gain over last  {gain_txt}",
        ]
        if len(self.plateau) > 1:
            lines.append(f"plateau         layers {self.plateau} scored within tolerance")
        if self.seed_layers:
            lines.append(
                f"seed agreement  {self.seed_agreement:.0%} "
                f"(layers chosen across seeds: {sorted(set(self.seed_layers))})"
            )

        conf = self.confidence
        if conf == "unmeasured":
            lines.append("confidence      not measured -- run with --n-seeds 5 or more")
        else:
            margin = self.peak_margin
            margin_txt = "n/a" if math.isnan(margin) else f"{margin:.2f}x"
            plateau_agr = self.plateau_agreement
            plateau_agr_txt = "n/a" if math.isnan(plateau_agr) else f"{plateau_agr:.0%}"
            spread = self.seed_spread
            spread_txt = "n/a" if math.isnan(spread) else f"{spread:.0%}"
            lines.append(
                f"confidence      {conf}   (plateau agreement {plateau_agr_txt}, "
                f"separation {margin_txt}, seed spread {spread_txt} of depth)"
            )

        lines.extend(f"note     {n}" for n in self.notes)
        return "\n".join(lines)

    def curve_plot(self) -> str:
        scores = [v for v in self.curve.values() if not math.isnan(v)]
        if not scores:
            return ""

        sigma = self.layer_sigma if self.seed_curves else {}
        has_sigma = any(math.isfinite(v) for v in sigma.values())

        def _bound(layer: int) -> float:
            sd = sigma.get(layer, 0.0)
            return sd if math.isfinite(sd) else 0.0

        if has_sigma:
            finite_items = [(layer, s) for layer, s in self.curve.items() if math.isfinite(s)]
            lo = min(s - _bound(layer) for layer, s in finite_items)
            hi = max(s + _bound(layer) for layer, s in finite_items)
        else:
            lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0

        header = f"layer performance ({self.metric}"
        header += f", +/- 1 sd across {len(self.seed_curves)} seeds)" if has_sigma else ")"
        lines = ["", header + ":"]
        plateau_set = set(self.plateau)
        for layer, s in sorted(self.curve.items()):
            if math.isnan(s):
                lines.append(f"  {layer:>3}      nan")
                continue
            sd_known = math.isfinite(sigma.get(layer, float("nan")))
            sd = sigma[layer] if sd_known else 0.0
            solid = "#" * max(1, int((s - lo) / span * 46))
            whisker = "-" * max(0, int(sd / span * 46)) if has_sigma else ""
            bar = solid + whisker
            if layer == self.best_layer:
                tag = "  <- best"
            elif layer == self.last_layer:
                tag = "  (last)"
            elif layer in plateau_set:
                tag = "  ~ plateau"
            else:
                tag = ""
            sd_txt = f"   +/-{sd:.4f}" if has_sigma and sd_known else ""
            lines.append(f"  {layer:>3} {s:+.4f} {bar}{sd_txt}{tag}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["curve"] = {str(k): v for k, v in self.curve.items()}
        d["seed_curves"] = [{str(k): v for k, v in c.items()} for c in self.seed_curves]
        d["layer_sigma"] = {str(k): v for k, v in self.layer_sigma.items()}
        for name in _DERIVED_FIELDS:
            d[name] = getattr(self, name)
        return d


def _argbest(curve: dict[int, float]) -> int:
    finite = {k: v for k, v in curve.items() if math.isfinite(v)}
    if not finite:
        raise RuntimeError("every layer scored NaN; the probe or labels are degenerate")
    return max(finite, key=finite.get)


def _plateau(curve: dict[int, float], peak: int, tolerance: float) -> list[int]:
    """Layers whose score is within ``tolerance`` (relative) of the peak."""
    best = curve[peak]
    if not math.isfinite(best):
        return [peak]
    band = tolerance * abs(best)
    keep = [layer for layer, s in curve.items() if math.isfinite(s) and s >= best - band]
    return sorted(keep) or [peak]


def _choose(curve: dict[int, float], tolerance: float) -> tuple[int, list[int]]:
    """Return ``(chosen, plateau)``.

    Among layers indistinguishable from the peak, take the **shallowest**: it
    scores the same but yields a smaller, faster truncated model, and it is a
    markedly more stable choice than a raw argmax over a flat region.
    """
    peak = _argbest(curve)
    plateau = _plateau(curve, peak, tolerance)
    return min(plateau), plateau


def select_layer(
    dataset: Dataset,
    train: np.ndarray,
    val: np.ndarray,
    *,
    model_name: str = "",
    probe: str = "knn",
    k: int = 10,
    n_seeds: int = 3,
    seed_fraction: float = 0.8,
    seed: int = 42,
    tolerance: float = 0.02,
) -> Result:
    """Probe every layer of ``train``/``val`` (each ``(n_states, N, D)``) and pick the best."""
    n_layers = train.shape[0] - 1
    candidates = list(range(n_layers + 1))

    def _score_layer(
        layer: int,
        rows: np.ndarray | None = None,
        val_rows: np.ndarray | None = None,
    ) -> float:
        tx, ty = _layer(train, layer), dataset.train_y
        if rows is not None:
            tx, ty = tx[rows], ty[rows]
        vx, vy = _layer(val, layer), dataset.val_y
        if val_rows is not None:
            vx, vy = vx[val_rows], vy[val_rows]
        pred = fit_predict(tx, ty, vx, vy, task=dataset.task, probe=probe, k=k)
        return score(pred, vy, dataset.task)

    curve = {layer: _score_layer(layer) for layer in candidates}
    best, plateau = _choose(curve, tolerance)

    # Stability check: does the choice survive re-drawing the training
    # subsample -- and, where there's enough of it, the validation subsample
    # too, since a fixed val set hides a large share of the run's real noise.
    n_train = len(dataset.train_y)
    n_val = len(dataset.val_y)
    resample_val = n_val >= 40

    def _draw_val_rows(rng: np.random.Generator, take: int) -> np.ndarray | None:
        for _ in range(3):
            rows = rng.choice(n_val, size=take, replace=False)
            if dataset.task != "classification" or len(np.unique(dataset.val_y[rows])) >= 2:
                return rows
        return None  # couldn't avoid losing a class; fall back to the full val set

    seed_layers: list[int] = []
    seed_curves: list[dict[int, float]] = []
    if n_seeds > 0 and n_train > 20:
        rng = np.random.default_rng(seed)
        take_train = max(10, int(n_train * seed_fraction))
        take_val = max(10, int(n_val * seed_fraction))
        for _ in range(n_seeds):
            rows = rng.choice(n_train, size=take_train, replace=False)
            val_rows = _draw_val_rows(rng, take_val) if resample_val else None
            sub = {layer: _score_layer(layer, rows, val_rows) for layer in candidates}
            seed_curves.append(sub)
            try:
                seed_layers.append(_choose(sub, tolerance)[0])
            except RuntimeError:
                continue

    agreement = (
        sum(layer == best for layer in seed_layers) / len(seed_layers)
        if seed_layers
        else float("nan")
    )

    notes: list[str] = []
    if len(plateau) > 1:
        notes.append(
            f"layers {plateau} are within {tolerance:.0%} of the peak; "
            f"took layer {best} for the smallest model at equal performance"
        )
    if best == n_layers:
        notes.append("the last layer won: truncation buys nothing on this dataset")
    if seed_layers and agreement < 0.5:
        notes.append(
            "layer choice is unstable across resampling -- raise sample, or "
            "treat the whole peak region as equally good"
        )

    return Result(
        model=model_name,
        dataset=dataset.name,
        task=dataset.task,
        probe=probe,
        best_layer=best,
        best_score=curve[best],
        last_layer=n_layers,
        last_layer_score=curve.get(n_layers, float("nan")),
        curve=curve,
        plateau=plateau,
        n_train=n_train,
        n_val=n_val,
        seed_layers=seed_layers,
        seed_agreement=agreement,
        notes=notes,
        tolerance=tolerance,
        seed_curves=seed_curves,
        n_train_available=dataset.n_train_available,
        n_val_available=dataset.n_val_available,
    )
