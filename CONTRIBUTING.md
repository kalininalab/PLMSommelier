# Contributing

Thanks for considering a contribution to PLMSommelier.

## Dev install

```bash
git clone https://github.com/kalininalab/PLMSommelier
cd PLMSommelier
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests -q                     # full suite (downloads a small HF checkpoint)
pytest tests -q -m "not weights"    # skip anything that needs model weights -- the default loop
pytest tests/test_select.py -q      # single file
```

Tests are marked `weights` (needs a real HF hub download) or `slow` (end-to-end).
CI runs `-m "not weights"` on every push/PR and the full suite, including
`weights`, on a weekly schedule -- so a contribution that only touches pure
logic (`data.py`, `select.py`) never needs to wait on a download, but
`model.py` changes should be checked against the full suite locally before
opening a PR.

## Lint and format

```bash
pre-commit install     # run ruff + basic hygiene checks on every commit
pre-commit run -a      # one-off pass over the whole repo
```

Or run the checks directly (this is what CI runs):

```bash
ruff check .
ruff format --check .
```

`ruff format .` / `ruff check --fix .` will fix most things automatically.

## Invariants that are easy to break silently

A few behaviors in `model.py` are load-bearing but not obvious from reading a
single function:

1. **The final norm is applied at every truncation point.** Every supported
   architecture applies a trailing norm (`emb_layer_norm_after`,
   `final_layer_norm`, `ln_f`) to the last block's output, so
   `hidden_states[i]` is raw for `i < L` but normed at `i = L`. Truncating to
   layer `k` must ship `final_norm(h_k)`, not raw `h_k` -- see
   `PLM._normalize_states`, checked numerically in `tests/test_model.py`.
2. **Model-specific quirks live in the `_QUIRKS` table, keyed on
   `config.model_type`**, not in per-family branches. T5, BERT and ALBERT
   (the ProtTrans family) all need space-joined residues and `[UZOB] -> X`;
   T5 additionally needs bfloat16 (fp16 overflows T5 activations). Add a new
   family by adding an entry, not a new code path.
3. ESM's `contact_head` is `Linear(n_layers * n_heads, 1)` -- truncation must
   resize it (`_resize_layer_dependent_heads`) or the saved checkpoint no
   longer matches its own config and fails to reload.
4. **Loading and reloading a model must use the same loader-class logic.**
   `load_hf_model` is the one place that decides between
   `AutoModelForTextEncoding`, `AutoModel`, and `AutoModelForCausalLM`; both
   the initial load and `tests/test_model.py`'s reload check call it.
   Reimplementing this choice elsewhere is how T5-encoder-only checkpoints
   ended up unreloadable -- a bare `AutoModel` resolves `T5Config` to the
   full seq2seq `T5Model`, not the `T5EncoderModel` that was actually saved.
5. **Pooling excludes special tokens via `return_special_tokens_mask`, not a
   calibrated layout.** This is a deliberate accuracy trade: a model whose
   direction/mode token is an *ordinary* added token (e.g. ProstT5's
   `<AA2fold>`) gets it included in the pooling mean -- one token out of
   hundreds, not worth a calibration subsystem to avoid.
6. **A sequence-length limit can come from a buffer, not just config.**
   `_length_limit` only sees `tokenizer.model_max_length` and (for
   absolute-position architectures) `config.max_position_embeddings`. ProGen2
   and ProtGPT2 report neither, but a longer-than-1024-token input still dies
   inside `torch.where` on a fixed-size causal-mask buffer -- `_causal_mask_limit`
   finds that ceiling empirically, the same way everything else in this module
   avoids hardcoding per architecture.
7. **A `trust_remote_code` truncation needs a self-consistent `auto_map`.**
   `_prepare_remote_code_for_save` resets it and points every entry at the
   files this save actually copies, as instance attributes on the model/config
   (`_auto_class`) -- never `register_for_auto_class`, which mutates the class
   and leaks into every future save of that architecture in the process. A
   stale hub-prefixed entry left over from the base checkpoint makes reload
   import the same config class twice under two different dynamic-module
   paths, which `AutoModel.register` then rejects as inconsistent.

## Pull requests

- Keep PRs focused; unrelated formatting churn makes review harder.
- Add or update tests for behavior changes.
- `ruff check .` and `pytest -q -m "not weights"` should both pass locally.
