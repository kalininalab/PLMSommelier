"""Load a HuggingFace PLM, embed every layer, and truncate to any depth.

The one subtlety worth reading before editing: ``hidden_states[i]`` for
``i < L`` is the **raw** block output, while ``hidden_states[L]`` has already
had the model's final norm applied. A model truncated to ``k`` blocks emits
``final_norm(h_k)``. So to make "probe layer k" and "ship a k-block model"
mean the same thing, the final norm is applied to every intermediate layer at
extraction time. See :func:`_normalize_states`.
"""

from __future__ import annotations

import copy
import importlib
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["PLM", "load_model", "embed_layers", "truncate", "save_truncated"]

# Where transformer blocks and the trailing norm live, per architecture.
# Probed in order; the first hit wins.
_BLOCK_OWNERS = ("", "encoder", "transformer", "model")
_BLOCK_ATTRS = ("layer", "block", "h", "layers")
_FINAL_NORM_ATTRS = (
    "emb_layer_norm_after",  # ESM-2
    "final_layer_norm",  # T5
    "ln_f",  # GPT-2 / ProGen2
    "layer_norm",
    "norm",
)
_DEPTH_KEYS = ("num_hidden_layers", "num_layers", "n_layer", "n_layers")

# ProtTrans-style tokenizers (T5, BERT, ALBERT) are trained on whitespace-
# separated single-residue tokens with rare residues folded into X; every
# family that shares that pretraining convention shares this pair.
_SPACED_RESIDUES = dict(
    space_join=True,
    residue_map=str.maketrans("UZOB", "XXXX"),
)

# Pretraining conventions HuggingFace's config does not record.
_QUIRKS: dict[str, dict[str, Any]] = {
    "t5": dict(
        _SPACED_RESIDUES,
        dtype=torch.bfloat16,  # T5 activations overflow in fp16
        encoder_only=True,  # loads T5EncoderModel; the decoder is never built
    ),
    "bert": dict(_SPACED_RESIDUES),  # ProtBert
    "albert": dict(_SPACED_RESIDUES),  # ProtAlbert
}

# Architectures that live in a third-party package rather than in transformers
# itself. These checkpoints carry no `auto_map`, so --trust-remote-code cannot
# reach them; the package registers them with the Auto classes on import.
# Keyed on config.model_type, like _QUIRKS. Values are (module, pip extra).
_REGISTRY_PACKAGES: dict[str, tuple[str, str]] = {
    "proteinbert": ("multimolecule", "multimolecule"),
}


def _preprocess_text(seq: str, quirks: dict[str, Any]) -> str:
    """Apply a model family's pretraining-time text convention to one sequence."""
    if quirks.get("residue_map"):
        seq = seq.translate(quirks["residue_map"])
    text = " ".join(seq) if quirks.get("space_join") else seq
    return quirks.get("prefix_text", "") + text


def _length_limit(tokenizer, config) -> int | None:
    """The longest input this architecture/tokenizer pair can safely accept.

    ``tokenizer.model_max_length`` is honored unless it's the HF placeholder
    for "unbounded" (a huge sentinel, historically ``1e30`` or ``int(1e12)``).
    ``config.max_position_embeddings`` is only a real ceiling for **absolute**
    position embeddings -- ESM-2 uses rotary embeddings and reports no such
    ceiling, and T5 maps ``max_position_embeddings`` from ``n_positions``
    (512) despite having no positional limit at all, so applying it
    unconditionally would silently truncate every ProtT5 input at 512
    residues. Where present, ``pad_token_id`` accounts for the RoBERTa/ESM
    convention of offsetting position ids by ``padding_idx + 1``.
    """
    limits = []

    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < int(1e6):
        limits.append(tok_limit)

    if getattr(config, "position_embedding_type", None) == "absolute":
        cfg_limit = getattr(config, "max_position_embeddings", None)
        if isinstance(cfg_limit, int) and cfg_limit > 0:
            pad_token_id = getattr(config, "pad_token_id", None)
            offset = pad_token_id + 1 if isinstance(pad_token_id, int) else 0
            limits.append(cfg_limit - offset)

    return min(limits) if limits else None


def _causal_mask_limit(model: nn.Module) -> int | None:
    """The longest sequence a fixed-size causal-mask buffer can attend over.

    Some architectures (ProGen2, ProtGPT2) report no usable limit from either
    ``tokenizer.model_max_length`` or ``config.max_position_embeddings`` --
    the latter is either absent or not an absolute-embedding config -- yet
    every attention block still registers a square boolean causal-mask buffer
    sized to a real, fixed context window (e.g. ``(1, 1, 1024, 1024)``). Feed
    it a longer input and it dies inside ``torch.where`` with a plain shape
    mismatch instead of a legible error. Detected empirically, like every
    other layout fact in this module, rather than hardcoded per architecture:
    any boolean buffer whose last two dims are equal and greater than one is
    such a mask, and the smallest one found is the real ceiling.
    """
    limit = None
    for buf in model.buffers():
        if buf.dtype != torch.bool or buf.ndim < 2:
            continue
        a, b = buf.shape[-2], buf.shape[-1]
        if a != b or a <= 1:
            continue
        limit = a if limit is None else min(limit, a)
    return limit


def _tokenization_is_degenerate(
    input_ids: torch.Tensor, pool_mask: torch.Tensor, unk_token_id: int | None
) -> bool:
    """Did the tokenizer collapse a real sequence into ~nothing informative?

    Two unambiguous failure shapes, both symptoms of feeding an unspaced
    sequence to a tokenizer trained on spaced residues (the ProtBert/
    ProtAlbert case): every pooled token is ``[UNK]``, or the whole sequence
    pooled to a single token. A BPE-style vocabulary that legitimately merges
    several residues into one token (e.g. ProGen2) still yields more than one
    pooled token, so it does not trip this.
    """
    pooled = input_ids[pool_mask]
    if pooled.numel() == 0:
        return False
    if unk_token_id is not None and bool((pooled == unk_token_id).all()):
        return True
    return pooled.numel() <= 1


def locate_blocks(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    """Find the ``ModuleList`` of transformer blocks and the module owning it."""
    for owner_name in _BLOCK_OWNERS:
        owner = model if not owner_name else getattr(model, owner_name, None)
        if owner is None:
            continue
        for attr in _BLOCK_ATTRS:
            blocks = getattr(owner, attr, None)
            if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                return owner, attr, blocks
    raise RuntimeError(f"cannot locate transformer blocks on {type(model).__name__}")


def locate_final_norm(owner: nn.Module) -> nn.Module | None:
    """Find the norm applied after the last block, if the architecture has one."""
    for attr in _FINAL_NORM_ATTRS:
        mod = getattr(owner, attr, None)
        if isinstance(mod, nn.Module):
            return mod
    return None


def _set_depth(config, depth: int) -> None:
    """Write the new block count to every depth key the config actually has."""
    written = False
    for key in _DEPTH_KEYS:
        if key in config.__dict__:
            setattr(config, key, depth)
            written = True
    if not written:  # configs exposing depth only via attribute_map
        for key in _DEPTH_KEYS:
            try:
                getattr(config, key)
            except AttributeError:
                continue
            setattr(config, key, depth)
            written = True
            break
    if not written:
        raise RuntimeError(f"cannot record depth on {type(config).__name__}")


def _install_get_head_mask() -> None:
    """transformers 5 dropped ``ModuleUtilsMixin.get_head_mask``, but v4-era
    remote code (e.g. ProGen2's ``modeling_progen.py``) still calls
    ``self.get_head_mask(...)`` on every forward pass. Restore the v4
    implementation on the base class -- a no-op if a future transformers
    brings the method back.
    """
    from transformers import PreTrainedModel

    if hasattr(PreTrainedModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            # per-layer, per-head mask
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=self.dtype)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    PreTrainedModel.get_head_mask = get_head_mask


def _install_tied_weights_backfill() -> None:
    """``all_tied_weights_keys`` is computed only in transformers 5's
    ``PreTrainedModel.post_init()`` -- but v4-era remote code's ``__init__``
    calls the older ``self.init_weights()`` instead and never calls
    ``post_init()``. ``from_pretrained`` unconditionally reads
    ``model.all_tied_weights_keys`` while finalising the load (e.g.
    ``ModuleUtilsMixin._move_missing_keys_from_meta_to_device``), so such a
    model crashes with ``AttributeError`` there. Backfill it from
    ``init_weights()`` itself, the same way ``post_init()`` would -- a no-op
    once it is already set (native v5 models, or a second call).
    """
    from transformers import PreTrainedModel

    if getattr(PreTrainedModel.init_weights, "_plmsommelier_patched", False):
        return

    orig_init_weights = PreTrainedModel.init_weights

    def init_weights(self):
        orig_init_weights(self)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)

    init_weights._plmsommelier_patched = True
    PreTrainedModel.init_weights = init_weights


@contextmanager
def _v4_style_init():
    """Build the model on a real device instead of meta, the way transformers 4 did.

    transformers 5 always initialises ``from_pretrained`` on the meta device
    and only materialises registered parameters/buffers from the checkpoint
    afterwards (``low_cpu_mem_usage=False`` is silently ignored). A plain
    tensor attribute a v4-era module computes in ``__init__`` -- e.g.
    ProGen2's ``ProGenAttention.scale_attn`` -- is never materialised and is
    left on the meta device, crashing the first real forward pass. Scoped to
    one ``from_pretrained`` call; native v5 checkpoints keep the fast
    meta-device path everywhere else.

    Once nothing is built on meta, ``_move_missing_keys_from_meta_to_device``
    -- whose whole job is moving meta-allocated tensors to a concrete device
    before they get randomly reinitialised -- becomes actively harmful: it
    still treats every non-persistent buffer (a checkpoint never carries
    those) as "missing" and overwrites it with `torch.empty_like`, i.e.
    uninitialised memory, clobbering an already-correct value like the
    causal-mask buffer computed moments ago in ``__init__``. Disabled for the
    same scope -- there is nothing left on meta for it to move.
    """
    from transformers import PreTrainedModel

    # The raw classmethod descriptor, not the bound method -- reassigning
    # `.__func__` back to the class would strip the `classmethod` wrapper and
    # break every subsequent access (including this function's own re-entry).
    orig_descriptor = vars(PreTrainedModel)["get_init_context"]
    orig = orig_descriptor.__func__

    @classmethod
    def patched(cls, *args, **kwargs):
        return [
            ctx
            for ctx in orig(cls, *args, **kwargs)
            if not (isinstance(ctx, torch.device) and ctx.type == "meta")
        ]

    orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device

    def no_op_move(self, *args, **kwargs):
        return None

    PreTrainedModel.get_init_context = patched
    PreTrainedModel._move_missing_keys_from_meta_to_device = no_op_move
    try:
        yield
    finally:
        PreTrainedModel.get_init_context = orig_descriptor
        PreTrainedModel._move_missing_keys_from_meta_to_device = orig_move


def load_hf_model(
    source,
    *,
    encoder_only: bool | None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    config=None,
):
    """Load a HF model, choosing the loader class that matches how it was saved.

    Shared by the initial load and the reload verification in the test suite,
    so the two can never disagree about which ``Auto*`` class to use -- that
    disagreement is what makes T5-encoder-only truncations unreloadable (a
    plain ``AutoModel`` resolves ``T5Config`` to the full seq2seq ``T5Model``,
    not the ``T5EncoderModel`` that was actually saved).
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForTextEncoding

    # transformers >= 5 treats a bare `torch_dtype=None` as `dtype="auto"`, which
    # adopts the checkpoint's own dtype instead of the fp32 that transformers 4.x
    # defaulted to -- pin the historical default explicitly so a reload's precision
    # doesn't depend on which transformers major happens to be installed.
    kw = dict(
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        torch_dtype=dtype if dtype is not None else torch.float32,
    )

    if encoder_only is None and config is not None:
        encoder_only = bool(getattr(config, "is_encoder_decoder", False))

    def _load():
        if encoder_only:
            # Loads T5EncoderModel, so the (discarded) decoder is never
            # materialised -- ProtT5-XL drops from ~3B to ~1.2B params.
            try:
                return AutoModelForTextEncoding.from_pretrained(source, **kw)
            except (KeyError, ValueError):
                model = AutoModel.from_pretrained(source, **kw)
                return getattr(model, "encoder", model)

        try:
            # Suppresses the randomly-initialised pooler that AutoModel would
            # otherwise attach to ESM -- random weights must never be shipped.
            return AutoModel.from_pretrained(source, add_pooling_layer=False, **kw)
        except TypeError:
            return AutoModel.from_pretrained(source, **kw)
        except ValueError:
            # Some checkpoints (e.g. ProGen2) register only AutoModelForCausalLM
            # in their auto_map, so AutoModel rejects the config outright. Fall
            # back to the causal-LM class and hand back its base transformer
            # stack -- the LM head is not needed for embeddings.
            lm = AutoModelForCausalLM.from_pretrained(source, **kw)
            return getattr(lm, lm.base_model_prefix, lm)

    if not trust_remote_code:
        return _load()

    # Only remote code is old enough to hit the v4/v5 gaps above.
    _install_get_head_mask()
    _install_tied_weights_backfill()
    with _v4_style_init():
        return _load()


class PLM:
    """A loaded HuggingFace protein language model, ready to embed and truncate."""

    def __init__(self, model_id: str, model, tokenizer, config, *, device, quirks):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.quirks = quirks

        owner, attr, blocks = locate_blocks(model)
        self.n_layers = len(blocks)
        self._final_norm = locate_final_norm(owner)
        limits = [
            lim for lim in (_length_limit(tokenizer, config), _causal_mask_limit(model)) if lim
        ]
        self._max_length = min(limits) if limits else None
        self._warned_truncation = False
        self._warm_up()

    def _preprocess(self, seq: str) -> str:
        return _preprocess_text(seq, self.quirks)

    def _encode(self, sequences: list[str]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Tokenize and return ``(model_inputs, pool_mask)``.

        ``pool_mask`` is what whole-protein pooling averages over: every real
        token that is not a special one (``<cls>``, ``<eos>``, padding, ...).
        """
        texts = [self._preprocess(s) for s in sequences]

        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=self._max_length is not None,
            max_length=self._max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        if self._max_length is not None and not self._warned_truncation:
            # A row that fills the whole budget with real (non-padding) tokens
            # was cut -- a legitimate sequence landing on exactly that many
            # tokens is possible but rare enough not to warrant a second,
            # untruncated tokenizer pass just to rule it out.
            if bool((enc["attention_mask"].sum(dim=1) >= self._max_length).any()):
                warnings.warn(
                    f"{self.model_id}: input sequence exceeds this architecture's "
                    f"{self._max_length}-token limit and will be truncated.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_truncation = True

        special = enc.pop("special_tokens_mask").bool()
        model_inputs = {
            k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
        }
        pool_mask = (enc["attention_mask"].bool() & ~special).to(self.device)
        return model_inputs, pool_mask

    def _warm_up(self) -> None:
        """One forward pass to establish ground truth about the model's outputs."""
        model_inputs, pool_mask = self._encode(["ACDEFGHIKLMNPQRSTVWY"])
        if _tokenization_is_degenerate(
            model_inputs["input_ids"], pool_mask, self.tokenizer.unk_token_id
        ):
            raise RuntimeError(
                f"{self.model_id}: tokenizing a 20-residue probe sequence produced "
                "no usable per-residue tokens (likely all [UNK], or the whole "
                "sequence collapsed to one token). This model's family probably "
                "needs a pretraining-convention entry in plmsommelier.model._QUIRKS "
                "(e.g. space-joined residues) -- see CONTRIBUTING.md."
            )
        with torch.no_grad():
            out = self.model(**model_inputs, output_hidden_states=True)
        self.n_states = len(out.hidden_states)
        self.hidden_size = int(out.hidden_states[0].shape[-1])

        # Is hidden_states[-1] already the post-final-norm state?
        last = getattr(out, "last_hidden_state", None)
        self._final_state_is_normed = bool(
            last is not None
            and last.shape == out.hidden_states[-1].shape
            and torch.allclose(last.float(), out.hidden_states[-1].float(), atol=1e-4)
        )

    def _normalize_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Make every layer equal what a model truncated there would emit."""
        states = list(hidden_states)
        if self._final_norm is not None:
            last = len(states) - 1
            for i in range(len(states)):
                if i == last and self._final_state_is_normed:
                    continue
                states[i] = self._final_norm(states[i])
        return torch.stack(states, dim=0)

    def forward(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run every sequence through the model. Returns ``(states, pool_mask)``.

        ``states`` is ``(n_states, B, T, D)``, already normalized so that
        ``states[k]`` equals a model truncated to ``k`` blocks' output.
        """
        model_inputs, pool_mask = self._encode(sequences)
        with torch.no_grad():
            out = self.model(**model_inputs, output_hidden_states=True)
            states = self._normalize_states(out.hidden_states)
        return states, pool_mask


def _arch_supports(device_cc: tuple[int, int], arches: list[str]) -> bool:
    """Does any compiled arch in ``arches`` run on a device of capability ``device_cc``?

    A cubin built for ``sm_X`` runs on a device of the *same major* version whose
    capability is >= X (minor-version forward compatibility); a ``compute_X``
    entry ships PTX the driver can JIT under the same rule.
    """
    major, minor = device_cc
    cc = major * 10 + minor
    for arch in arches:
        _, _, digits = arch.partition("_")
        if not digits.isdigit():
            continue
        target = int(digits)
        if target // 10 == major and cc >= target:
            return True
    return False


def _cuda_is_usable() -> bool:
    """True only if a CUDA device exists *and* this torch build has kernels for it.

    ``torch.cuda.is_available()`` is true for any visible device, including one
    whose compute capability was never compiled into the installed wheel (e.g.
    an sm_61 card on a cu13 build). Selecting it then dies at the first kernel
    launch with ``no kernel image is available for execution on the device``,
    so auto-detection has to check the arch list too.
    """
    if not torch.cuda.is_available():
        return False
    try:
        device_cc = torch.cuda.get_device_capability(0)
        arches = list(torch.cuda.get_arch_list())
    except Exception:
        return False
    if not arches:
        # ROCm and other unusual builds report no arch list; trust is_available().
        return True
    if _arch_supports(device_cc, arches):
        return True
    warnings.warn(
        f"CUDA device 0 ({torch.cuda.get_device_name(0)}, compute capability "
        f"{device_cc[0]}.{device_cc[1]}) is not supported by this torch build "
        f"(compiled for {', '.join(arches)}); falling back to CPU. Pass "
        "device='cuda' (--device cuda) to use it anyway, or install a torch "
        "build with kernels for this GPU.",
        RuntimeWarning,
        stacklevel=3,
    )
    return False


def _select_device() -> torch.device:
    """Auto-detect ``cuda`` > ``mps`` > ``cpu``, in order of likely speed."""
    mps = getattr(torch.backends, "mps", None)
    if _cuda_is_usable():
        return torch.device("cuda")
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bf16_is_usable(device: torch.device) -> bool:
    """MPS has no (or, on older torch, incomplete) bfloat16 support. CUDA's
    ``is_bf16_supported`` is the authority to defer to rather than a compute
    capability guess of our own -- but only when asked about *real* hardware
    support. On torch >= 2.6 the no-argument call defaults to
    ``including_emulation=True`` and reports ``True`` on pre-Ampere cards
    (e.g. Turing, sm_75) that merely emulate bf16 in software: the model
    loads fine and halves its memory footprint, but matmuls get no
    tensor-core speedup, silently making a "bf16" run look like a size
    optimization when it's actually running at roughly fp32 speed. Ask for
    ``including_emulation=False`` explicitly; older torch that doesn't know
    the kwarg falls back to the plain call.
    """
    if device.type == "mps":
        return False
    if device.type != "cuda":
        return True
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        pass  # older torch that doesn't know the kwarg -- fall back below
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _resolve_dtype(
    dtype: torch.dtype | None, quirks: dict[str, Any], device: torch.device, model_id: str
) -> torch.dtype:
    """Pick a dtype, then downgrade it if the device can't actually run it.

    The T5 family's dtype quirk exists to dodge fp16 overflow, so fp32 is the
    right fallback rather than trying fp16 and hitting the same issue.
    """
    resolved = dtype or quirks.get("dtype")
    if resolved is None:
        resolved = torch.float16 if device.type == "cuda" else torch.float32
    if resolved == torch.bfloat16 and not _bf16_is_usable(device):
        warnings.warn(
            f"{model_id}: bfloat16 is not supported on {device}; using float32 instead.",
            stacklevel=2,
        )
        resolved = torch.float32
    return resolved


def _load_config(model_id: str, **common: Any):
    """``AutoConfig.from_pretrained``, rescued for architectures registered by
    a third-party package (see ``_REGISTRY_PACKAGES``) rather than by transformers.
    """
    from transformers import AutoConfig, PretrainedConfig

    try:
        return AutoConfig.from_pretrained(model_id, **common)
    except ValueError:
        # Unknown model_type. Reading the raw config dict needs no registered
        # config class, so it works even though AutoConfig just failed.
        config_dict, _ = PretrainedConfig.get_config_dict(model_id, **common)
        model_type = config_dict.get("model_type")
        entry = _REGISTRY_PACKAGES.get(model_type)
        if entry is None:
            raise
        module, extra = entry
        try:
            importlib.import_module(module)  # registers with the Auto classes
        except ImportError as exc:
            raise RuntimeError(
                f"{model_id}: model type '{model_type}' is provided by the "
                f"'{module}' package, which is not installed. Install it with "
                f"`pip install plmsommelier[{extra}]`."
            ) from exc
        return AutoConfig.from_pretrained(model_id, **common)


def load_model(
    model_id: str,
    *,
    device: str | None = None,
    cache_dir: str | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
) -> PLM:
    """Load ``model_id`` (a HuggingFace id or local path) ready for embedding."""
    from transformers import AutoTokenizer

    device = _select_device() if device is None else torch.device(device)

    common = dict(trust_remote_code=trust_remote_code, cache_dir=cache_dir)
    config = _load_config(model_id, **common)
    quirks = dict(_QUIRKS.get(getattr(config, "model_type", ""), {}))
    if "prostt5" in model_id.lower():
        # <AA2fold> is an ordinary added token giving the translation
        # direction; every other T5-family quirk still applies.
        quirks["prefix_text"] = "<AA2fold> "

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    except Exception:
        # Some sentencepiece Unigram models (e.g. ProtT5) cannot be
        # fast-converted on every `tokenizers` version; the slow, pure-Python
        # tokenizer loads the same checkpoint fine.
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **common)

    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.unk_token
        if fallback is None:
            raise RuntimeError(f"{model_id}: tokenizer has no pad, eos, or unk token")
        tokenizer.pad_token = fallback

    resolved_dtype = _resolve_dtype(dtype, quirks, device, model_id)

    hf_model = load_hf_model(
        model_id,
        encoder_only=quirks.get("encoder_only"),
        dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        config=config,
    )
    hf_model = hf_model.to(device).eval()

    return PLM(model_id, hf_model, tokenizer, config, device=device, quirks=quirks)


def _batches(lengths: list[int], max_tokens: int, max_size: int) -> list[list[int]]:
    """Length-sorted batches under a token budget.

    Sorting by length keeps padding overhead low and bounds peak activation
    memory far better than a fixed batch size.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        nxt_max = max(cur_max, lengths[i])
        if cur and (len(cur) + 1 > max_size or nxt_max * (len(cur) + 1) > max_tokens):
            out.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = nxt_max
    if cur:
        out.append(cur)
    return out


def embed_layers(
    plm: PLM,
    sequences: list[str],
    *,
    max_tokens_per_batch: int = 8192,
    max_batch_size: int = 64,
    progress: bool = False,
) -> np.ndarray:
    """Mean-pool every layer's representation of each sequence.

    Returns ``(n_states, len(sequences), hidden_size)`` float16.
    """
    lengths = [len(s) for s in sequences]
    longest = max(lengths, default=0)
    if longest > max_tokens_per_batch:
        # `_batches` only bounds the *product* of batch size and max length in
        # a batch -- a single sequence longer than the budget still gets its
        # own batch of size 1, unbounded. Attention memory is O(length^2), so
        # this is the one shape `load_dataset`'s `max_seq_len` cap can't catch
        # for callers who build `sequences` themselves.
        warnings.warn(
            f"{plm.model_id}: a sequence of {longest} residues exceeds "
            f"max_tokens_per_batch={max_tokens_per_batch}; it will still be embedded "
            "alone in its own batch and may exceed available memory.",
            RuntimeWarning,
            stacklevel=2,
        )

    out = np.zeros((plm.n_states, len(sequences), plm.hidden_size), dtype=np.float16)
    batches = _batches(lengths, max_tokens_per_batch, max_batch_size)

    iterator = batches
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="embedding", unit="batch")
        except ImportError:
            pass

    for group in iterator:
        states, pool_mask = plm.forward([sequences[i] for i in group])
        for slot, seq_i in enumerate(group):
            valid = pool_mask[slot]
            if not bool(valid.any()):
                raise RuntimeError(f"sequence at row {seq_i}: nothing to pool over")
            pooled = states[:, slot][:, valid].float().mean(dim=1)  # (S, D)
            out[:, seq_i] = pooled.to(torch.float16).cpu().numpy()

    return out


def truncate(plm: PLM, layer: int) -> nn.Module:
    """Return a standalone model whose ``last_hidden_state`` is layer ``layer``.

    ``layer`` indexes the same way as :func:`embed_layers`: 0 is the embedding
    output (a model with no transformer blocks), ``plm.n_layers`` is the full
    model.
    """
    if not 0 <= layer <= plm.n_layers:
        raise ValueError(f"layer must be in [0, {plm.n_layers}], got {layer}")

    # Deep-copying the full model first (then discarding blocks) needs a
    # second full-size resident copy at its peak. Instead, temporarily shrink
    # the *original* model's block list to just the retained blocks,
    # deepcopy that, then restore the original list unconditionally.
    owner, attr, blocks = locate_blocks(plm.model)
    full_blocks = blocks
    try:
        setattr(owner, attr, nn.ModuleList(list(blocks)[:layer]))
        truncated = copy.deepcopy(plm.model)
    finally:
        setattr(owner, attr, full_blocks)

    _set_depth(truncated.config, layer)
    _resize_layer_dependent_heads(truncated, layer)
    truncated.eval()
    return truncated


def _resize_layer_dependent_heads(model: nn.Module, layer: int) -> None:
    """Shrink auxiliary heads whose input width is a function of depth.

    ESM's ``contact_head`` consumes the attention maps of every layer, so its
    regression is ``Linear(n_layers * n_heads, 1)``. Left unchanged, a
    deep-copied head no longer matches the truncated config and
    ``from_pretrained`` fails with a size mismatch on reload. Attention maps
    are ordered layer-major, so the first ``layer * n_heads`` columns are
    exactly the retained layers' weights.
    """
    head = getattr(model, "contact_head", None)
    regression = getattr(head, "regression", None)
    if regression is None:
        return
    n_heads = getattr(model.config, "num_attention_heads", None)
    if not isinstance(n_heads, int):
        return
    keep = layer * n_heads
    if keep == regression.in_features:
        return
    resized = nn.Linear(
        keep,
        regression.out_features,
        bias=regression.bias is not None,
        device=regression.weight.device,
        dtype=regression.weight.dtype,
    )
    with torch.no_grad():
        resized.weight.copy_(regression.weight[:, :keep])
        if regression.bias is not None:
            resized.bias.copy_(regression.bias)
    head.regression = resized
    if hasattr(head, "in_features"):
        head.in_features = keep


def _is_remote_code(obj: Any) -> bool:
    """Was ``obj``'s class dynamically loaded via ``trust_remote_code``?

    Transformers imports such classes under the synthetic ``transformers_modules``
    package (see ``get_class_from_dynamic_module``); anything else is a class
    transformers ships itself.
    """
    return type(obj).__module__.startswith("transformers_modules")


def _prepare_remote_code_for_save(model: nn.Module, auto_class: str) -> None:
    """Make a remote-code truncated model saveable *and reloadable*.

    A checkpoint like ProGen2 only registers ``AutoModelForCausalLM`` in its
    ``auto_map`` -- there is no ``AutoModel`` entry, because the hub checkpoint
    is the causal-LM class. ``truncate()`` hands back the *base* transformer
    (no LM head), so plain ``model.save_pretrained`` writes no ``auto_map``
    entry a ``trust_remote_code`` loader could use for it at all.

    Worse, the moment anything in this process (e.g. a prior test, or this
    very function on a previous call) has triggered
    ``ProGenConfig.register_for_auto_class()`` -- which ``AutoConfig.from_pretrained``
    of a *local* directory does automatically -- that permanently flips a
    class-level flag. Every following ``save_pretrained`` of *any* instance of
    that config class then writes a self-referential ``auto_map`` that points
    at the local directory's own files, while any stale hub-prefixed entry
    left over from the base checkpoint (``AutoModelForCausalLM`` here) still
    points at the *hub's* copy of the same module. Reload then imports
    ``ProGenConfig`` twice -- once per path -- as two distinct dynamic module
    objects, and ``AutoModel.register`` rejects the mismatch with "The model
    class you are passing has a `config_class` attribute that is not
    consistent with the config class you passed".

    The fix is to make every ``auto_map`` entry in the saved directory
    resolve to the *same* local files. Setting ``_auto_class`` as an
    **instance** attribute (never calling the classmethod
    ``register_for_auto_class``, which would repeat the same mistake at
    class scope) makes ``save_pretrained`` copy this model's own module file
    (plus its relative imports, i.e. the config module) into the output
    directory and record a bare local reference for it; starting from an
    empty ``auto_map`` drops every stale hub-prefixed entry inherited from
    the base checkpoint, including any class truncation removed entirely
    (the LM head).
    """
    if not _is_remote_code(model):
        return
    config = model.config
    config.auto_map = {}
    config._auto_class = "AutoConfig"
    model._auto_class = auto_class


def _preprocessing_note(quirks: dict[str, Any]) -> str:
    """Describe this family's input convention, built from the quirks actually
    set rather than hardcoded per family -- stays accurate as `_QUIRKS` grows.

    Empty string for a family with no text-preprocessing quirks (dtype and
    encoder_only don't change what the caller has to do to the input text).
    """
    parts = []
    if quirks.get("prefix_text"):
        parts.append(f"prefixed with {quirks['prefix_text']!r}")
    if quirks.get("space_join"):
        parts.append('residues space-joined (e.g. "A C D E" instead of "ACDE")')
    residue_map = quirks.get("residue_map")
    if residue_map:
        src = "".join(chr(k) for k, v in residue_map.items() if v is not None)
        dst = chr(next(v for v in residue_map.values() if v is not None))
        parts.append(f"residues [{src}] mapped to {dst!r}")
    return "; ".join(parts)


def _model_card(
    result, plm: PLM, layer: int, model_cls: str, out: Path, *, trust_remote_code: bool
) -> str:
    gain = result.gain_over_last
    gain_txt = "n/a" if gain != gain else f"{gain:+.1%}"
    seed_agreement_txt = (
        "n/a"
        if result.seed_agreement != result.seed_agreement
        else (f"{result.seed_agreement:.0%}")
    )
    load_kwargs = ", trust_remote_code=True" if trust_remote_code else ""

    registry_entry = _REGISTRY_PACKAGES.get(getattr(plm.config, "model_type", ""))
    registry_import = (
        f"import {registry_entry[0]}  # registers the architecture\n" if registry_entry else ""
    )
    registry_note = (
        f"\nReloading needs the `{registry_entry[0]}` package: "
        f"`pip install plmsommelier[{registry_entry[1]}]`.\n"
        if registry_entry
        else ""
    )

    preprocessing = _preprocessing_note(plm.quirks)
    preprocessing_note = (
        f"\n**This model family needs preprocessed input**: {preprocessing}. Feeding "
        "raw sequences instead of applying this will silently produce degraded "
        "embeddings -- see `plmsommelier.model._preprocess_text`.\n"
        if preprocessing
        else ""
    )

    return f"""---
library_name: transformers
tags: [protein-language-model, plmsommelier, truncated]
base_model: {plm.model_id}
---

# {plm.model_id} truncated to {layer} of {plm.n_layers} layers

Produced by [`plmsommelier`](https://github.com/kalininalab/PLMSommelier), which
picks the most informative layer of a protein language model for a specific
downstream dataset instead of defaulting to the last one.
{registry_note}
```python
{registry_import}from transformers import {model_cls}, AutoTokenizer
model = {model_cls}.from_pretrained("{out}"{load_kwargs})
tok = AutoTokenizer.from_pretrained("{out}"{load_kwargs})
```
{preprocessing_note}
| | |
|---|---|
| dataset | `{result.dataset}` |
| task | {result.task} |
| score at layer {layer} | **{result.best_score:.4f}** |
| score at layer {plm.n_layers} (the usual choice) | {result.last_layer_score:.4f} |
| gain over the last layer | **{gain_txt}** |
| seed agreement | {seed_agreement_txt} |
| confidence | {result.confidence} |

Layer choice is dataset-specific -- re-run `plmsommelier` for a new dataset.
"""


def save_truncated(plm: PLM, result, out_dir: str | Path) -> Path:
    """Truncate to ``result.best_layer`` and write a loadable model directory."""
    layer = result.best_layer
    out = Path(out_dir)

    model = truncate(plm, layer)

    encoder_only = plm.quirks.get("encoder_only")
    if encoder_only is None:
        encoder_only = bool(getattr(plm.config, "is_encoder_decoder", False))
    model_cls = "AutoModelForTextEncoding" if encoder_only else "AutoModel"
    trust_remote_code = _is_remote_code(model)
    _prepare_remote_code_for_save(model, model_cls)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    plm.tokenizer.save_pretrained(out)

    (out / "README.md").write_text(
        _model_card(result, plm, layer, model_cls, out, trust_remote_code=trust_remote_code)
    )
    (out / "plmsommelier.json").write_text(
        json.dumps(
            {
                **result.to_dict(),
                "base_model": plm.model_id,
                "layers_kept": layer,
                "layers_total": plm.n_layers,
                "preprocessing": _preprocessing_note(plm.quirks) or None,
            },
            indent=2,
            default=str,
        )
    )
    return out
