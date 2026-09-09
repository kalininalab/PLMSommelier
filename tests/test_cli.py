"""CLI argument resolution, and one real end-to-end run.

``cli.py`` derives every ``suggest`` flag from ``suggest_layer``'s own
function signature and numpydoc docstring via cyclopts, so
``app.parse_args``/``app.meta.parse_args`` resolve and bind arguments without
ever calling the underlying function -- these tests need no model weights.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from plmsommelier.cli import _render_confidence_warning, app, launcher, main, suggest_layer
from plmsommelier.select import Result

FULL_ARGV = [
    "suggest",
    "--data",
    "x.csv",
    "--model",
    "m",
    "--label-col",
    "foo",
    "--task",
    "classification",
    "--probe",
    "lr",
    "--k",
    "7",
    "--sample",
    "111",
    "--max-seq-len",
    "333",
    "--n-seeds",
    "7",
    "--seed",
    "99",
    "--tolerance",
    "0.05",
    "--device",
    "cpu",
    "--cache-dir",
    "/tmp/c",
    "--out",
    "/tmp/o",
    "--no-progress",
]


class TestSuggestArgumentResolution:
    def test_full_argv_resolves_every_flag(self):
        """Every non-default value on the CLI reaches suggest_layer's binding."""
        command, bound, ignored = app.parse_args(FULL_ARGV)
        assert command is suggest_layer
        assert not ignored
        args = bound.arguments
        assert str(args["data"]) == "x.csv"
        assert args["model"] == "m"
        assert args["label_col"] == ["foo"]
        assert args["task"] == "classification"
        assert args["probe"] == "lr"
        assert args["k"] == 7
        assert args["sample"] == 111
        assert args["max_seq_len"] == 333
        assert args["n_seeds"] == 7
        assert args["seed"] == 99
        assert args["tolerance"] == 0.05
        assert args["device"] == "cpu"
        assert args["cache_dir"] == "/tmp/c"
        assert args["out"].as_posix() == "/tmp/o"
        assert args["progress"] is False

    def test_label_col_flag_resolves_to_a_list(self):
        command, bound, ignored = app.parse_args(
            ["suggest", "--data", "x.csv", "--model", "m", "--label-col", "foo"]
        )
        assert bound.arguments["label_col"] == ["foo"]


class TestSuggestDefaultsGuard:
    def test_every_keyword_param_and_default_matches_cyclopts_own_resolution(self):
        """The drift guard: checked against cyclopts' own resolved argument
        collection, not against ``suggest_layer``'s signature that
        ``bound.apply_defaults()`` itself reads from."""
        sig = inspect.signature(suggest_layer)
        keyword_params = {
            name: param
            for name, param in sig.parameters.items()
            if param.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and name not in ("data", "model")
        }

        argument_collection = app["suggest"].assemble_argument_collection()
        by_param_name = {arg.field_info.name: arg for arg in argument_collection}

        missing = keyword_params.keys() - by_param_name.keys()
        assert not missing, f"parameters with no CLI flag: {missing}"

        for name, param in keyword_params.items():
            arg = by_param_name[name]
            assert arg.field_info.default == param.default, (
                f"{name}: cyclopts resolved default {arg.field_info.default!r} "
                f"!= suggest_layer's signature default {param.default!r}"
            )


class TestMaxSeqLenWiring:
    """`--max-seq-len` must reach `load_dataset`, with `0` meaning "disabled"
    (`None`) rather than a literal zero-residue cap."""

    def _patch(self, monkeypatch):
        import plmsommelier.cli as cli_mod

        captured = {}
        fake_ds = SimpleNamespace(train_seqs=[], val_seqs=[])

        def _fake_load_dataset(data, **kwargs):
            captured.update(kwargs)
            return fake_ds

        monkeypatch.setattr(cli_mod, "load_dataset", _fake_load_dataset)
        monkeypatch.setattr(cli_mod, "load_model", lambda *a, **kw: SimpleNamespace(model_id="m"))
        monkeypatch.setattr(cli_mod, "embed_layers", lambda *a, **kw: None)
        monkeypatch.setattr(cli_mod, "select_layer", lambda *a, **kw: "result")
        return captured

    def test_explicit_value_reaches_load_dataset(self, monkeypatch):
        captured = self._patch(monkeypatch)
        result = suggest_layer("x.csv", "m", max_seq_len=333)
        assert captured["max_seq_len"] == 333
        assert result == "result"

    def test_zero_disables_the_cap(self, monkeypatch):
        captured = self._patch(monkeypatch)
        suggest_layer("x.csv", "m", max_seq_len=0)
        assert captured["max_seq_len"] is None


class TestToleranceWiring:
    """`--tolerance` must reach `select_layer` -- it wasn't wired at all before."""

    def _patch(self, monkeypatch):
        import plmsommelier.cli as cli_mod

        captured = {}
        fake_ds = SimpleNamespace(train_seqs=[], val_seqs=[])

        monkeypatch.setattr(cli_mod, "load_dataset", lambda data, **kw: fake_ds)
        monkeypatch.setattr(cli_mod, "load_model", lambda *a, **kw: SimpleNamespace(model_id="m"))
        monkeypatch.setattr(cli_mod, "embed_layers", lambda *a, **kw: None)

        def _fake_select_layer(*a, **kw):
            captured.update(kw)
            return "result"

        monkeypatch.setattr(cli_mod, "select_layer", _fake_select_layer)
        return captured

    def test_explicit_value_reaches_select_layer(self, monkeypatch):
        captured = self._patch(monkeypatch)
        result = suggest_layer("x.csv", "m", tolerance=0.1)
        assert captured["tolerance"] == 0.1
        assert result == "result"

    def test_default_value_reaches_select_layer(self, monkeypatch):
        captured = self._patch(monkeypatch)
        suggest_layer("x.csv", "m")
        assert captured["tolerance"] == 0.02


def _low_confidence_result(**overrides) -> Result:
    """A hand-built Result whose stats deterministically land on "low"."""
    base = dict(
        model="m",
        dataset="d",
        task="regression",
        probe="knn",
        best_layer=2,
        best_score=0.9,
        last_layer=4,
        last_layer_score=0.25,
        curve={0: 0.1, 1: 0.2, 2: 0.9, 3: 0.3, 4: 0.25},
        plateau=[2],
        n_train=100,
        n_val=50,
        seed_layers=[0, 4, 1],
        seed_agreement=0.0,
        seed_curves=[
            {0: 0.5, 1: 0.4, 2: 0.45, 3: 0.3, 4: 0.6},
            {0: 0.3, 1: 0.6, 2: 0.4, 3: 0.35, 4: 0.5},
            {0: 0.4, 1: 0.35, 2: 0.3, 3: 0.6, 4: 0.45},
        ],
        n_train_available=500,
        n_val_available=200,
    )
    base.update(overrides)
    return Result(**base)


class TestConfidenceWarningBlock:
    def test_low_confidence_prints_to_stderr_not_stdout(self, capsys):
        _render_confidence_warning(_low_confidence_result(), quiet=False)
        out, err = capsys.readouterr()
        assert out == ""
        assert "LOW CONFIDENCE" in err
        assert "raise --sample" in err

    def test_quiet_suppresses_the_block(self, capsys):
        _render_confidence_warning(_low_confidence_result(), quiet=True)
        out, err = capsys.readouterr()
        assert out == err == ""

    def test_unmeasured_does_not_print_the_block(self, capsys):
        r = _low_confidence_result(seed_layers=[], seed_curves=[], seed_agreement=float("nan"))
        assert r.confidence == "unmeasured"
        _render_confidence_warning(r, quiet=False)
        out, err = capsys.readouterr()
        assert out == err == ""

    def test_moderate_or_high_does_not_print_the_block(self, capsys):
        r = _low_confidence_result(
            plateau=[0, 1, 2, 3, 4],
            seed_layers=[2, 2, 2],
            seed_curves=[
                {0: 0.85, 1: 0.87, 2: 0.9, 3: 0.86, 4: 0.84},
                {0: 0.84, 1: 0.86, 2: 0.9, 3: 0.85, 4: 0.83},
                {0: 0.86, 1: 0.88, 2: 0.9, 3: 0.87, 4: 0.85},
            ],
        )
        assert r.confidence != "low"
        _render_confidence_warning(r, quiet=False)
        out, err = capsys.readouterr()
        assert out == err == ""


class TestJsonWriteErrorHandling:
    """Regression test: `--json` writing used to sit outside the launcher's
    try/except, so an OSError while writing the JSON file (its parent path
    is unusable, say) escaped as a raw traceback instead of the same
    `error: ...` handling every other caught exception gets."""

    def _patch(self, monkeypatch, tmp_path):
        import plmsommelier.cli as cli_mod

        result = Result(
            model="m",
            dataset="d",
            task="regression",
            probe="knn",
            best_layer=0,
            best_score=0.5,
            last_layer=0,
            last_layer_score=0.5,
            curve={0: 0.5},
            plateau=[0],
            n_train=10,
            n_val=5,
        )
        monkeypatch.setattr(cli_mod, "load_model", lambda *a, **kw: SimpleNamespace(model_id="m"))
        monkeypatch.setattr(
            cli_mod, "load_dataset", lambda *a, **kw: SimpleNamespace(train_seqs=[], val_seqs=[])
        )
        monkeypatch.setattr(cli_mod, "embed_layers", lambda *a, **kw: None)
        monkeypatch.setattr(cli_mod, "select_layer", lambda *a, **kw: result)

    def test_unwritable_json_path_is_caught_not_raised(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        blocking_file = tmp_path / "not_a_directory"
        blocking_file.write_text("x")
        bad_json = blocking_file / "out.json"  # parent isn't a directory

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--json",
                    str(bad_json),
                    "suggest",
                    "--data",
                    "x.csv",
                    "--model",
                    "m",
                    "--no-progress",
                ]
            )
        assert exc_info.value.code == 1


class TestInvalidLiteralChoices:
    @pytest.mark.parametrize("flag,bad_value", [("--task", "nonsense"), ("--probe", "xyz")])
    def test_invalid_choice_exits_nonzero(self, flag, bad_value):
        with pytest.raises(SystemExit) as exc_info:
            app.parse_args(["suggest", "--data", "x.csv", "--model", "m", flag, bad_value])
        assert exc_info.value.code != 0


class TestLauncherMetaFlags:
    """``--json``/``--quiet`` live on the meta-app's own ``launcher``, not on
    ``suggest_layer`` -- resolved via ``app.meta.parse_args`` so nothing is
    executed."""

    def test_json_and_quiet_resolve_distinctly_from_suggest_args(self):
        command, bound, ignored = app.meta.parse_args(
            ["--json", "out.json", "--quiet", "suggest", "--data", "x.csv", "--model", "m"]
        )
        assert command is launcher
        assert not ignored
        assert str(bound.arguments["json_out"]) == "out.json"
        assert bound.arguments["quiet"] is True
        assert bound.arguments["tokens"] == ("suggest", "--data", "x.csv", "--model", "m")

    def test_json_and_quiet_default_when_omitted(self):
        command, bound, ignored = app.meta.parse_args(
            ["suggest", "--data", "x.csv", "--model", "m"]
        )
        assert command is launcher
        bound.apply_defaults()
        assert bound.arguments["json_out"] is None
        assert bound.arguments["quiet"] is False


@pytest.mark.weights
@pytest.mark.slow
def test_end_to_end_suggest_and_truncate(tmp_path):
    """Real run: a toy CSV, a real (tiny) model, a loadable truncated result."""
    n = 60
    df = pd.DataFrame(
        {
            "ID": [f"P{i}" for i in range(n)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"[: 6 + i % 12] for i in range(n)],
            "label": [(i % 12) * 0.1 for i in range(n)],
        }
    )
    data_path = tmp_path / "toy.csv"
    df.to_csv(data_path, index=False)
    out_dir = tmp_path / "truncated"
    json_path = tmp_path / "result.json"

    # cyclopts wraps an int return in sys.exit(), even on success (code 0).
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--json",
                str(json_path),
                "suggest",
                "--data",
                str(data_path),
                "--model",
                "facebook/esm2_t6_8M_UR50D",
                "--out",
                str(out_dir),
                "--n-seeds",
                "0",
                "--no-progress",
            ]
        )
    assert exc_info.value.code == 0

    payload = json.loads(json_path.read_text())
    assert "best_layer" in payload
    # --n-seeds 0 means the stability check never ran: "unmeasured", not "low".
    assert payload["confidence"] == "unmeasured"

    from transformers import AutoModel

    model = AutoModel.from_pretrained(out_dir, add_pooling_layer=False)
    assert model.config.num_hidden_layers == payload["best_layer"]
