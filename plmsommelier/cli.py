"""Command line interface, built on cyclopts.

Every flag under ``suggest`` is derived from ``suggest_layer``'s own function
signature and numpydoc docstring -- nothing is redeclared here. This module
only owns presentation (rendering the result) and error handling, via a
``@app.meta.default`` launcher that wraps flags shared across every command
(``--json``, ``--quiet``, ``--traceback``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from plmsommelier import __version__
from plmsommelier.data import Task, load_dataset
from plmsommelier.model import embed_layers, load_model, save_truncated
from plmsommelier.select import Result, select_layer

app = App(
    name="plmsommelier",
    version=__version__,
    help=(
        "Pick the most informative layer of a protein language model for your "
        "dataset, and get back a truncated model."
    ),
)

_CAUGHT_EXCEPTIONS = (ValueError, RuntimeError, MemoryError, OSError, IndexError)


def suggest_layer(
    data: Path,
    model: str,
    *,
    task: Task | None = None,
    # narrowed to `list[str] | None`: cyclopts cannot resolve `str | list[str]`.
    label_col: list[str] | None = None,
    probe: Literal["knn", "lr"] = "knn",
    k: int = 10,
    sample: float | None = 5000,
    max_seq_len: int = 2000,
    n_seeds: int = 5,
    seed: int = 42,
    tolerance: float = 0.02,
    device: str | None = None,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    out: Path | None = None,
    progress: bool = True,
) -> Result:
    """Find the best layer of ``model`` for ``data``; optionally save it truncated.

    Parameters
    ----------
    data : Path
        CSV with ID, sequence and a label column.
    model : str
        a HuggingFace model id or a local path.
    task : Task | None, optional
        default: inferred from the label column where unambiguous (multiple
        columns, non-numeric values, exactly two values, or non-whole-numbered
        floats). A column of whole numbers with more than two values is
        genuinely ambiguous -- including one pandas only stored as float
        because of an unrelated blank row -- pass this explicitly as
        'classification' or 'regression'.
    label_col : list[str] | None, optional
        override label column detection.
    probe : Literal["knn", "lr"], optional
        layer probe to fit.
    k : int, optional
        neighbours for the kNN probe.
    sample : float | None, optional
        budget for the whole dataset, split between train and validation.
        A value >= 1 is an absolute row count (e.g. 5000); a value in
        (0, 1) is a fraction of the whole dataset (e.g. 0.1 for 10%). Pass
        None to disable subsampling entirely.
    max_seq_len : int, optional
        drop sequences longer than this many residues before splitting or
        subsampling (0 disables the cap). Protects against a single outlier
        protein OOMing the run -- attention memory is quadratic in length.
    n_seeds : int, optional
        resampling stability checks -- redraws train (and, where there's
        enough of it, validation) rows this many times and re-scores every
        layer, feeding the ``confidence`` verdict and its remedies. Raising
        this *measures* confidence more precisely; it does not by itself
        raise it -- more data (``--sample``), a lower-variance probe, or a
        genuinely sharper curve are what do that.
    seed : int, optional
        random seed.
    tolerance : float, optional
        relative score band (of the peak) within which layers are treated as
        tied -- the "plateau". Widening it merges more layers into the tied
        region; narrowing it makes the peak choice stricter.
    device : str | None, optional
        device to load the model onto; default: cuda, else mps, else cpu.
    cache_dir : str | None, optional
        HuggingFace cache directory.
    trust_remote_code : bool, optional
        needed for checkpoints that ship custom modeling code.
    out : Path | None, optional
        directory to write the truncated model into.
    progress : bool, optional
        show progress while extracting embeddings.

    Returns
    -------
    Result
        the chosen layer, its score, and the full per-layer curve.
    """
    plm = load_model(model, device=device, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    ds = load_dataset(
        data,
        task=task,
        label_col=label_col,
        sample=sample,
        max_seq_len=max_seq_len or None,
        seed=seed,
    )

    train = embed_layers(plm, ds.train_seqs, progress=progress)
    val = embed_layers(plm, ds.val_seqs, progress=progress)

    result = select_layer(
        ds,
        train,
        val,
        model_name=plm.model_id,
        probe=probe,
        k=k,
        n_seeds=n_seeds,
        seed=seed,
        tolerance=tolerance,
    )
    if out is not None:
        save_truncated(plm, result, out)
    return result


app.command(suggest_layer, name="suggest")


def _render_confidence_warning(result: Result, *, quiet: bool) -> None:
    """A loud, low-noise warning on stderr when confidence is low.

    Fires only on "low" -- not "unmeasured", which just means the stability
    check wasn't run and isn't itself evidence of a bad pick. Prints to
    stderr so stdout (and --json) stay parseable, and always exits 0: the
    result is a valid best guess, merely an uncertain one.
    """
    if quiet or result.confidence != "low":
        return
    sys.stdout.flush()  # keep banner after the summary when stdout is redirected
    rule = "-" * 66
    lines = [rule, "LOW CONFIDENCE in the selected layer", ""]
    plateau_agr = result.plateau_agreement
    if math.isfinite(plateau_agr):
        lines.append(
            f"  seed agreement  {result.seed_agreement:.0%} exact, "
            f"{plateau_agr:.0%} within plateau "
            f"(picks: {sorted(set(result.seed_layers))})"
        )
    margin = result.peak_margin
    if math.isfinite(margin):
        lines.append(f"  peak margin     {margin:.2f}x over the best layer outside the plateau")
    if result.n_train_available is not None and result.n_val_available is not None:
        lines.append(
            f"  data used       {result.n_train}+{result.n_val} of "
            f"{result.n_train_available}+{result.n_val_available} available rows"
        )
    lines += [
        "",
        f"  Layer {result.best_layer} is still the best single guess, but a rerun on a",
        "  different subsample may well pick a different layer.",
        "",
        "  What to do about it:",
    ]
    lines.extend(f"    - {r}" for r in result.remedies)
    lines.append(rule)
    print("\n".join(lines), file=sys.stderr)


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    json_out: Annotated[Path | None, Parameter(name="--json")] = None,
    quiet: bool = False,
    traceback: bool = False,
) -> int:
    """
    Parameters
    ----------
    json_out : Path | None
        also write the result as JSON.
    quiet : bool
        suppress the rendered summary/curve output.
    traceback : bool
        let exceptions propagate with their full traceback instead of being
        caught and rendered as ``error: ...``.
    """
    try:
        command, bound, _ = app.parse_args(tokens)
        out = bound.arguments.get("out")
        result = command(*bound.args, **bound.kwargs)

        if not quiet:
            print(result.summary())
            print(result.curve_plot())
            _render_confidence_warning(result, quiet=quiet)
        if out is not None and not quiet:
            print(f"\ntruncated model written to {out}")
            print(f"  load it as shown in {out}/README.md")
        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    except _CAUGHT_EXCEPTIONS as exc:
        if traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return app.meta(argv)


if __name__ == "__main__":
    raise SystemExit(main())
