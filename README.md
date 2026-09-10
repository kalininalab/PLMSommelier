# PLMSommelier

[![PyPI](https://img.shields.io/pypi/v/plmsommelier)](https://pypi.org/project/plmsommelier/)
[![Python](https://img.shields.io/pypi/pyversions/plmsommelier)](https://pypi.org/project/plmsommelier/)
[![CI](https://github.com/kalininalab/PLMSommelier/actions/workflows/ci.yml/badge.svg)](https://github.com/kalininalab/PLMSommelier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The last layer of a protein language model is almost never the best one. This
tool finds the layer that is, and hands you back a truncated model.**

Implements the tool described in [*Task- and dataset-specific information
in protein language models*](https://arxiv.org/abs/2608.12090), which probed 13 PLMs across 15
downstream tasks and found the deepest layer won in only **~20%** of cases.

## Install

```bash
uv tool install plmsommelier   # isolated, no venv to manage yourself
# or: pipx install plmsommelier
# or: pip install plmsommelier   # into a virtualenv
```

<details>
<summary>CPU vs. GPU torch (Linux)</summary>

PLMSommelier itself is CUDA-agnostic -- it never links CUDA directly, so any
torch build works. But on Linux, plain `pip install plmsommelier` resolves
torch's *default* index, which is a CUDA build pulling several GB of `nvidia-*`
packages even on a machine with no GPU. Install torch yourself first if that's
not what you want:

```bash
# CPU-only (much smaller download; also what you want on a machine with no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install plmsommelier

# GPU: pick the wheel matching your driver, then install as usual
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA 12
pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA 13
pip install plmsommelier
```

macOS wheels are CPU-only already (Apple Silicon uses `mps` automatically at
runtime instead), so the plain `pip install plmsommelier` is correct there --
no separate torch step needed. If a `cuXXX` install fails or a model refuses
to use your GPU, check that your driver actually supports the CUDA version
that wheel targets; a driver too old for the chosen wheel is the usual cause.

</details>

## Quickstart

Use the `examples/fluorescence_sample.csv` that ships in the repo (a 500-row subsample of
the TAPE fluorescence benchmark).

```bash
plmsommelier suggest examples/fluorescence_sample.csv facebook/esm2_t6_8M_UR50D \
    --task regression --out ./my-esm-truncated
```

`data` and `model` are positional (`DATA MODEL`); `--data`/`--model` work
identically if you prefer named flags.

```
model    facebook/esm2_t6_8M_UR50D
dataset  fluorescence_sample  (regression, pearson, knn probe)
data     400 train / 100 val

best layer      0 of 6 (0% depth)   pearson = 0.4238
last layer      6               pearson = 0.2369
gain over last  +78.9%
seed agreement  100% (layers chosen across seeds: [0])
confidence      moderate   (plateau agreement 100%, separation 1.51x, seed spread 0% of depth)

layer performance (pearson, +/- 1 sd across 5 seeds):
    0 +0.4238 ########################################-----   +/-0.0537  <- best
    1 +0.3764 ####################################---------   +/-0.1012
    2 +0.3770 ####################################-----   +/-0.0634
    3 +0.1572 ###############------   +/-0.0736
    4 +0.0432 ####----   +/-0.0519
    5 +0.2014 ###################-------   +/-0.0824
    6 +0.2369 #######################------   +/-0.0710  (last)
```

The run returns `./my-esm-truncated` - a normal HuggingFace model directory containing the truncated model. It loads anywhere the original did, runs faster, and
scores at least as well on your task:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./my-esm-truncated")
tok = AutoTokenizer.from_pretrained("./my-esm-truncated")
```

Any HuggingFace protein language model works — just pass its id to `model`.
Checkpoints that ship custom modeling code need `--trust-remote-code`. See
[Extending to a custom PLM](#extending-to-a-custom-plm) below for models that
need more than that.

## Python API

If you prefer to use the library directly, the same workflow is available in Python:

```python
from plmsommelier import embed_layers, load_dataset, load_model, save_truncated, select_layer

plm = load_model("facebook/esm2_t6_8M_UR50D")
ds = load_dataset("examples/fluorescence_sample.csv", task="regression")

train_embeddings = embed_layers(plm, ds.train_seqs)
val_embeddings = embed_layers(plm, ds.val_seqs)

result = select_layer(ds, train_embeddings, val_embeddings, model_name=plm.model_id)
print(result.summary())  # the block printed by the CLI
print(result.curve_plot())  # the ASCII per-layer curve
result.to_dict()  # what --json writes out

save_truncated(plm, result, "./my-esm-truncated")
```

`Result` (`plmsommelier.select.Result`) carries the full per-layer curve
(`result.curve`), the per-seed scores behind the stability check
(`result.seed_curves`, `result.layer_sigma`), and the `confidence` verdict --
see [Confidence and what to do about it](#confidence-and-what-to-do-about-it).

## Input format

A CSV with `sequence` and a label column:

| column | required | notes |
|---|---|---|
| `sequence` | yes | amino acid sequence |
| `label` / `labels` / `Y` | yes | the target (checked in that order, or pass `--label-col`) |
| `ID` | no | not used by `plmsommelier`; keep it for your own bookkeeping if you like |
| `split` | no | `train` / `valid` (or `val`); generated if absent |

`--task` (`regression`, `classification` or `multi-label`) is inferred from
the label column when that's unambiguous.
If that fails, pass `--task` explicitly.

`--max-seq-len` (default 2000 residues) drops longer sequences before
splitting or subsampling, so a single outlier protein can't end up alone in
an unbounded batch -- attention memory is quadratic in length. Pass `0` to
disable it.

Device is auto-detected -- CUDA, else Apple Silicon's `mps`, else CPU -- and
can be forced with `--device cpu` / `--device cuda` / `--device mps`. A
visible CUDA device only counts if the installed torch build actually ships
kernels for it: an older card against a build that
dropped support for it would otherwise crash on the first forward pass with
`no kernel image is available for execution on the device` instead of just
running on the CPU. Auto-detect falls through with a warning in that case;
pass `--device cuda` explicitly to force it anyway.

## Extending to a custom PLM

There is no model registry -- `model` takes any HuggingFace id or local path,
and works with zero configuration for most architectures. Reach for the
options below only as far as your checkpoint actually needs, in order of
how rare each one is:

1. **Nothing.** `load_model` (`plmsommelier/model.py`) is architecture-agnostic:
   it finds the transformer blocks and final norm by attribute name, detects
   the sequence-length ceiling from the tokenizer/config/causal-mask buffer,
   and picks the right `Auto*` loading class. Most encoder PLMs on the Hub
   just work.

2. **`--trust-remote-code`** for checkpoints that ship custom modeling code
   in their own repo (an `auto_map` in `config.json`). Same flag `transformers`
   itself uses.

3. **A pretraining-convention entry in `_QUIRKS`** (`plmsommelier/model.py`),
   keyed on `config.model_type`, for models whose config doesn't record how
   they were actually pretrained -- HuggingFace has no field for "residues
   are space-separated tokens", for instance. Supported keys:

   | key | effect |
   |---|---|
   | `space_join` | join residues with spaces before tokenizing (ProtTrans-style) |
   | `residue_map` | a `str.translate` table applied to the sequence first (e.g. rare residues -> `X`) |
   | `prefix_text` | text prepended before tokenizing (e.g. a mode/direction token) |
   | `dtype` | force this torch dtype instead of the auto-detected default |
   | `encoder_only` | force `AutoModelForTextEncoding` (`True`) vs. `AutoModel`/`AutoModelForCausalLM` (`False`), overriding the `is_encoder_decoder` config check |

   T5/BERT/ALBERT (the ProtTrans family) are the worked example already in
   the table:

   ```python
   "t5": dict(
       space_join=True,
       residue_map=str.maketrans("UZOB", "XXXX"),
       dtype=torch.bfloat16,   # T5 activations overflow in fp16
       encoder_only=True,      # loads T5EncoderModel; the decoder is never built
   ),
   ```

   Add a family the same way: an entry keyed on its `model_type`
   (`AutoConfig.from_pretrained(your_model).model_type` tells you the key),
   with only the keys it actually needs.

4. **A `_REGISTRY_PACKAGES` entry** for architectures that live in a
   third-party package rather than in `transformers` itself -- these
   checkpoints carry no `auto_map`, so `--trust-remote-code` can't reach them;
   the package must be imported first so it registers itself with the `Auto*`
   classes. `multimolecule/proteinbert` is the current example: install
   `plmsommelier[multimolecule]`, and `model.py` imports `multimolecule`
   before loading whenever `config.model_type == "proteinbert"`. Add a new
   package the same way: `"model_type": ("import_name", "pip_extra_name")`,
   plus an extra in `pyproject.toml` if it isn't already installed alongside
   `plmsommelier`.

5. **No fork needed for a one-off.** `_QUIRKS` and `_REGISTRY_PACKAGES` are
   plain module-level dicts -- from your own script, `import plmsommelier.model
   as m; m._QUIRKS["your_model_type"] = {...}` before calling `load_model`
   works without touching this repo. Treat that as a private escape hatch,
   though (the leading underscore is deliberate) -- if it's a real,
   reusable family, a PR adding it to the table is the better home for it.

Whatever route you take, verify a new family the way `tests/test_model.py`
verifies the built-in ones: the truncation round-trip (`save_truncated` then
reload reproduces the same layer's output), sequence-length detection not
silently truncating valid input, and the final-norm invariant --
see [CONTRIBUTING.md](CONTRIBUTING.md)'s invariants list for what each of
those actually guards against.

### Tested models

Checkpoints below were loaded and probed end-to-end (`load_model` +
`embed_layers`) against the real weights, not just read from the code — either
by the repo's own real-weight test fixtures (`tests/test_model.py`) or by a
manual smoke test. Anything not listed still has a good chance of working —
`load_model` is architecture-agnostic — it just hasn't been verified here yet.

| Model | HuggingFace ID | Status |
|---|---|---|
| ESM-2 | `facebook/esm2_t6_8M_UR50D` (+ larger) | ✅ confirmed (test suite) |
| ESM-1b | `facebook/esm1b_t33_650M_UR50S` | ✅ confirmed (manual) |
| IgBert | `Exscientia/IgBert` | ✅ confirmed (manual) |
| ProGen2 | `hugohrban/progen2-small` (+ larger) | ✅ confirmed (test suite) |
| proteinbert (multimolecule) | `multimolecule/proteinbert` | ✅ confirmed (test suite) |


ProtBert, ProtAlbert, ProtT5, Ankh, ProstT5, and ProtGPT2 are handled by name
in `_QUIRKS` but have no real-weight test coverage yet — treat them as
likely-to-work, not confirmed.

## Confidence and what to do about it

Every run redraws the training data (and, where there's enough of it, the
validation data) `--n-seeds` times and re-scores every layer, to check
whether the chosen layer survives resampling. That feeds a `confidence`
verdict -- `high`, `moderate`, `low` or `unmeasured` -- printed as part of the summary and
recorded in `--json` output.

The verdict is the *worst* of three independent signals, not an average, so
one strong number can't cover another weak one:

- **seed agreement** -- how often resampling lands back inside the same
  plateau.
- **peak margin** -- how many (paired) standard deviations separate the
  plateau from the best layer outside it, across resamples.
- **seed spread** -- how far, on average, resampled picks land from the
  chosen layer, as a fraction of the network's depth.

A `low` verdict prints a warning block, which will probably require you either to increase the sample size.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev install and workflow, and
[RELEASING.md](RELEASING.md) for how a release is cut and published.

```bash
pip install -e ".[dev]"
pytest tests -q                      # full suite
pytest tests -q -m "not weights"     # no model downloads needed
```

## Citation

```bibtex
@article{joeres2026taskdataset,
  title  = {Task- and dataset-specific information in protein language models},
  author = {Joeres, Roman and Senatorov, Ilya and Kolchina, Anastasia and
            Klakow, Dietrich and Kalinina, Olga V.},
  journal = {arXiv preprint arXiv:2608.12090},
  year   = {2026}
}
```

See [CITATION.cff](CITATION.cff) for the software-citation form.

## License

MIT — see [LICENSE](LICENSE).
