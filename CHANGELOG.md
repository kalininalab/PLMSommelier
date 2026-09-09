# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Raised the declared minimum torch to `>=2.4` (from `>=2.0`). This corrects an
  under-declared floor rather than adding a new restriction: transformers 5.x
  (`transformers>=5.9` has been required since 1.0.0) disables torch entirely
  below 2.4, so torch 2.0-2.3 never actually worked with this package.

## [1.0.0] - 2026-09-09

### Fixed

- `--trust-remote-code` checkpoints written against transformers 4.x (ProGen2,
  and any other v4-era remote code) failed to load at all under
  transformers 5: the removed `ModuleUtilsMixin.get_head_mask` crashed the
  first forward pass, and transformers 5's meta-device `from_pretrained`
  init left `__init__`-time plain-tensor attributes (e.g. ProGen2's
  `scale_attn`) stranded on the meta device. `load_hf_model` now restores
  the v4 `get_head_mask` and builds v4-era remote code on a real device
  instead of meta, scoped to `trust_remote_code=True` loads only.

## [0.1.0] - 2026-09-08

Initial public release. Extracted from the paper reproduction repository
([Old-Shatterhand/SMTB2025_solution](https://github.com/Old-Shatterhand/SMTB2025_solution))
as its own package.

### Added

- A `confidence` verdict (`high`/`moderate`/`low`/`unmeasured`) on `Result`,
  derived from three resampling-stability signals -- plateau-aware seed
  agreement, a paired peak-margin ratio, and seed-pick spread -- combined by
  taking the *worst* of the three rather than an average. Printed in
  `suggest`'s summary, written into `--json` output and the truncated
  model's `README.md`, and recorded verdict-only in the model card. A `low`
  verdict also prints a warning block (stderr, suppressed by `--quiet`,
  never a nonzero exit) with concrete, run-specific remedies -- see the
  README's "Confidence and what to do about it".
- `Result.seed_curves`/`layer_sigma` -- the per-seed, per-layer scores behind
  the stability check are now kept instead of discarded, so `curve_plot()`
  can render a `+/-1 sd` error bar per layer.
- `Dataset.n_train_available`/`n_val_available` -- row counts before
  `--sample` subsampling, so the confidence remedies can tell "raise
  --sample" apart from "you're already using every row in the file".
- `--tolerance` on `suggest` -- `select_layer` already accepted it, but
  `suggest_layer` never passed it through, so there was no way to set it
  from the CLI.

### Changed

- The stability check (`--n-seeds`) now also resamples the validation set,
  not just training rows, when there's enough of it (`n_val >= 40`) -- a
  fixed val set was hiding a large share of a run's real noise. Its default
  rose `3` -> `5`: more seeds only measure stability more precisely, cost is
  a handful of cheap re-fits against embeddings already computed, and 3
  draws quantized agreement to quarters. Combined, both changes mean
  **`seed_agreement`'s numeric value shifts for identical inputs** compared
  to earlier runs (its exact-match definition is unchanged; a `< 0.5`
  "unstable" heuristic used in this project's own benchmarking still applies
  to it unmodified) -- historical benchmark data computed with the old
  estimator was not re-derived.

### Removed (simplification pass)

Cut the package to roughly a fifth of its size, keeping only what the tool
actually needs: read a CSV, infer the task, embed every layer, probe each
one, report the best layer, ship the truncated model.

- The model registry (`registry.py`, `userreg.py`, `plmsommelier models
  add`/`remove`/`list`/`info`, `~/.config/plmsommelier/models.toml`). Any
  HuggingFace protein language model works by passing its id to `--model`;
  the handful of pretraining conventions HuggingFace doesn't publish (T5
  wants space-joined residues, `[UZOB]->X`, bfloat16) now live in a small
  `model_type`-keyed table in `model.py` instead of a per-checkpoint catalog.
- Residue-level probing and the empirical tokenizer calibration
  (`calibrate.py`) that existed to support it. Embedding is now a plain
  mean-pool over the attention mask minus special tokens.
- Runtime self-verification of truncation (`verify_truncation`,
  `TruncationCheck`, the reload check in `save_truncated`). The guarantee --
  a truncated model reproduces the probed layer, and reloads faithfully --
  is now enforced in `tests/test_model.py` instead of on every real run.
- `TASK_METRICS`/`HIGHER_IS_BETTER`/the metric menu (`metric.py`). Every task
  now has exactly one metric: Pearson's r for regression, MCC for
  classification and multi-label.
- The `coarse` layer schedule, DeepLoc2.0 column detection, and the
  CUDA-arch-compatibility probe (`_cuda_is_usable`); device selection is now
  a plain `cuda if available else cpu`.
- `AmbiguousTaskError`/`CalibrationError`/`ResidueLevelUnsupported` and most
  of the package's dataclasses (`ModelSpec`, `TokenLayout`, `PresetInfo`,
  `ModelInfo`, `EncodedBatch`, `LayerStack`, `EmbeddingSet`, `Split`,
  `Registered`). Plain `ValueError`/`RuntimeError` and two remaining
  dataclasses (`Dataset`, `Result`) do the same job.

### Fixed

- ProtBert/ProtAlbert (and any other BERT/ALBERT-family ProtTrans checkpoint)
  collapsed every sequence to a single `[UNK]` token, since only the T5
  family's `_QUIRKS` entry space-joined residues. Added `bert`/`albert`
  entries, and a warm-up check that raises for any family whose tokenization
  is still unambiguously degenerate.
- Task inference could silently flip from raising (ambiguous integer column)
  to guessing `"regression"` depending on whether an unrelated, already-
  dropped row happened to have a blank label -- pandas upcasts an otherwise-
  integer column to `float64` the moment any row is missing. `infer_task` now
  sees the label column's missing values before they're dropped, and no
  longer trusts a NaN-driven float upcast alone.
- No `truncation`/`max_length` was passed to the tokenizer, so an
  absolute-position-embedding architecture (e.g. ESM-1b) could index out of
  range on a sequence longer than its position table. `_encode` now truncates
  to the architecture's real limit and warns once when it does.
- The KNN layer probe compared raw, unnormalized embeddings across layers,
  letting a few high-variance ("rogue") feature dimensions in one layer's
  activations dominate the distance calculation regardless of how
  informative that layer actually was. Both probes now fit a `StandardScaler`
  on train before the estimator. Reported per-layer scores shift slightly.
- A single label column holding list values, delimited multi-value strings
  (e.g. `"0,1,1,0"`), or per-residue annotations used to be silently treated
  as ordinary multi-class classification over an explosion of near-unique
  "classes". `load_dataset`/`infer_task` now raise, naming the column and the
  supported multi-label shape.
- `_length_limit` had no way to know ProGen2's/ProtGPT2's real 1024-token
  context window (neither `tokenizer.model_max_length` nor
  `config.max_position_embeddings` reports it for these architectures), so a
  longer input crashed inside the model with a raw tensor-shape mismatch
  instead of being truncated with a warning. `_causal_mask_limit` now detects
  the ceiling from each attention block's fixed-size causal-mask buffer.
- Saving a `trust_remote_code` truncation (e.g. ProGen2) a second time in the
  same process failed to reload: the first reload made transformers register
  the config class for its own auto-loading globally, so the next save wrote
  a self-referential `auto_map` for some entries while a stale hub-prefixed
  entry survived for others, and reload imported the same config class twice
  under two different module paths. `save_truncated` now resets `auto_map`
  and points every entry at the files it actually copies.
- `_bf16_is_usable` trusted `torch.cuda.is_bf16_supported()`, which on
  torch >= 2.6 reports `True` on pre-Ampere cards that only *emulate* bf16 in
  software -- ProtT5 (whose `_QUIRKS` entry requests bf16) loaded at half the
  memory with none of the speedup, misrepresenting throughput. Now asks for
  `including_emulation=False` explicitly.
- `load_dataset` required an `ID` column nothing downstream reads, and only
  recognized `label`/`labels` label columns -- most of this repo's own sample
  data (`Y`-labeled, no `ID`) couldn't be loaded without a separate
  normalization pass. `ID` is no longer required, and `Y` joins the label
  fallback chain.
- A single very long sequence (a real dataset had one at 13k, another at 35k
  residues) formed its own unbounded batch regardless of
  `max_tokens_per_batch`, since only the *product* of batch size and max
  length in a batch was bounded -- attention memory is quadratic in length, so
  one outlier could OOM a run no matter how conservative the token budget.
  `load_dataset` now takes `max_seq_len` (default 2000) and drops longer
  sequences before the split and subsample; `embed_layers` also warns once if
  a caller-supplied sequence still exceeds the batch budget on its own.

### Changed

- `binary` and `multi-class` tasks merged into one `classification` task --
  the probes and MCC scoring already treated them identically, so the split
  only bought branching.
- Five source modules instead of thirteen: `data.py`, `model.py`, `select.py`,
  `cli.py`, `__init__.py`.

### Added

- `plmsommelier suggest` -- given a HF checkpoint and a labeled CSV, finds the
  best layer via kNN/logistic-regression probing across seeds and saves a
  truncated, HuggingFace-loadable model.
- `plmsommelier models add`/`models remove` -- register a custom checkpoint
  (with optional layer/param hints and loading options) into a TOML file at
  `~/.config/plmsommelier/models.toml` or a project-local `plmsommelier.toml`,
  so it resolves like any built-in preset.
- Parameter-count hints for every builtin preset (`models list` now shows a
  `params` column alongside the existing layer hint), on the same
  documentation-only contract as the layer hints -- the authoritative count
  always comes from measuring the loaded checkpoint.
- `plmsommelier models` -- lists benchmarked presets or introspects any
  checkpoint's layer count and token layout.
- Presets for ESM-2, ProtT5, ProstT5, Ankh, ProGen2, and ProtGPT2, plus alias
  resolution for the paper's old model names.
- Empirical token-layout calibration (`calibrate.py`) instead of hardcoded
  per-architecture offsets.
- Numerically verified truncation: every saved model is checked against the
  representation the probe actually scored before it's written to disk.
- `Dataset.classes` -- for `binary`/`multi-class` tasks, the label column's
  original values in code order, so `y`'s integer codes can be mapped back to
  their real names.

### Changed

- CLI rewritten on [cyclopts](https://cyclopts.readthedocs.io/); several
  interface details changed as a result:
  - `plmsommelier models` -> `plmsommelier models list` (browse presets) /
    `plmsommelier models info MODEL` (introspect a checkpoint).
  - `suggest`'s `-k` short flag is gone; only `--k` remains.
  - `suggest`'s resampling-count flag was renamed `--seeds` -> `--n-seeds`
    (matches `suggest_layer`'s `n_seeds` parameter). Note that `--seed` (the
    RNG seed) is a different, pre-existing flag, so a stale `--seeds` in a
    script will fail with "Did you mean --seed?", which is not the right fix.
  - `--quiet` used to suppress both the rendered output and the progress bar;
    it now only suppresses the summary/curve. `--no-progress` independently
    controls the progress bar.
  - `--json` moved from a subcommand-local flag to a top-level (meta-app)
    flag; it can now be given at any position on the command line, before,
    after, or interleaved with the subcommand and its other flags.
  - `plmsommelier --version` now exists.
  - Every `suggest` flag, default, and choice is now generated directly from
    `plmsommelier.api.suggest_layer`'s signature and docstring instead of
    being hand-duplicated in the CLI, so `suggest --help` and the Python API
    can no longer drift apart.
  - `suggest`'s `DATA`/`MODEL` and `models info`'s `MODEL` are now also usable
    as positional arguments, not just as `--data`/`--model` flags (e.g.
    `plmsommelier suggest my.csv facebook/esm2_t6_8M_UR50D` works).
- `infer_task` no longer guesses on whole-number label columns (the old
  cardinality/density thresholds -- ``distinctness < 0.5``, ``<= 2000``
  classes, a warning above 50 -- are gone). Multiple columns, non-numeric
  values, exactly two distinct values, and float dtypes are still resolved
  automatically; an integer column with more than two values now raises
  `AmbiguousTaskError` naming the column and asking for an explicit
  `task=`/`--task`. An explicitly-supplied `task` is now validated against
  the data instead of being trusted blindly.
- `suggest_layer`'s `label_col` parameter now takes a list of column names
  instead of a bare string. (This union-to-list narrowing was needed to fix a
  cyclopts crash introduced earlier in this same rewrite -- the old,
  already-shipped argparse CLI's `--label-col` flag was never affected, and
  no shipped user is impacted. This is a heads-up rather than a migration
  note: Python callers passing a bare string should check the current
  signature.)

### Fixed

Found by an end-to-end benchmark run (`ANALYSIS.md`) before the first release;
`plmsommelier suggest --out ...` could not save a model at all on a GPU
machine, its entire deliverable.

- `save_truncated` failed on every GPU run: the reloaded model used to verify
  the save was never moved to the adapter's device, so the numeric check was
  fed CUDA tensors against a CPU (and often wrong-dtype) model.
- T5-encoder-only presets (`ankh_base`, `ankh_large`, `prott5`, `prostt5`)
  could never pass that reload check even on CPU: the reload path used a bare
  `AutoModel`, which resolves to the full seq2seq `T5Model` rather than the
  `T5EncoderModel` that was actually saved. Loading and reloading now share
  one function (`adapters.hf.load_hf_model`) so they can't disagree.
- `prott5`/`prostt5`'s tokenizer failed to load on newer `tokenizers`
  versions, which can't fast-convert their sentencepiece Unigram model;
  `HFAdapter.load()` now retries with the slow tokenizer.
- `progen2_small`/`medium`/`large` failed to load entirely: their
  `config.json` registers only `AutoModelForCausalLM`, which plain `AutoModel`
  rejects. Loading now falls back to the causal-LM class and keeps its base
  transformer stack.
- `protgpt2` (and any other subword-tokenized model) produced all-NaN
  whole-protein embeddings: pooling was derived from `residue_index`, which is
  only populated for residue-level tokenizers, so the pooling mask was
  silently all-`False`. Extraction now carries an explicit pooling mask
  populated for every model, and raises instead of silently averaging over
  zero elements if a mask still comes out empty.
- Tokenizers with no pad token (e.g. GPT-2-family, used by ProtGPT2 and
  ProGen2) crashed on the adapter's own warm-up forward pass; `load()` now
  falls back to `eos_token`/`unk_token` and warns.
- `HFAdapter.truncate()` deep-copied the full model before dropping blocks,
  needing a second full-size resident copy at its peak; it now shrinks the
  block list before copying, so peak memory is roughly "full model + kept
  blocks" instead of "two full models". `save_truncated` also frees its
  in-memory truncated copy before reloading the saved one, instead of holding
  three copies of a checkpoint at once.
- A non-numeric (e.g. string) `binary`/`multi-class` label column crashed
  with `invalid literal for int()`: `infer_task` correctly identified the
  task, but nothing then mapped the string values to integer codes before
  `y.astype(int)`. `load_dataset` now encodes such columns to `0..C-1` and
  records the original names on `Dataset.classes`.
- Automatic device selection trusted `torch.cuda.is_available()` alone, which
  is true for any visible CUDA device even if the installed torch build has no
  kernels for its compute capability (e.g. an older sm_61 GPU against a wheel
  compiled for sm_75+). That crashed the first forward pass with `CUDA error:
  no kernel image is available for execution on the device` instead of using
  the CPU. `load_adapter` (and therefore every command that doesn't pass
  `--device` explicitly) now checks the device's compute capability against
  the build's arch list and falls back to CPU with a warning when it's
  unsupported; passing `--device cuda` explicitly still fails loudly as before.
