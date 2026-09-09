"""Loading, embedding and truncating a real HuggingFace PLM.

The loader-class fallback chain is tested without downloading weights (it's
pure dispatch logic); everything else needs the tiny ``esm2_t6_8M`` checkpoint
and is marked ``weights``.
"""

from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from plmsommelier.model import (  # noqa: E402
    _REGISTRY_PACKAGES,
    _cuda_is_usable,
    _load_config,
    _resolve_dtype,
    _select_device,
    load_hf_model,
)


class _Sentinel:
    """Stands in for a loaded model; identity is all these tests check."""


class TestLoaderClassChain:
    def test_encoder_only_prefers_auto_model_for_text_encoding(self, monkeypatch):
        import transformers

        text_encoding = _Sentinel()
        monkeypatch.setattr(
            transformers.AutoModelForTextEncoding,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: text_encoding),
        )

        def _unexpected(cls, *a, **kw):
            raise AssertionError("AutoModel should not run when AutoModelForTextEncoding works")

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", classmethod(_unexpected))

        assert load_hf_model("fake/repo", encoder_only=True) is text_encoding

    def test_causal_lm_only_config_falls_back_from_auto_model(self, monkeypatch):
        """Pins the ProGen2 case: config.json registers only AutoModelForCausalLM."""
        import transformers

        class _LM:
            base_model_prefix = "transformer"

            def __init__(self):
                self.transformer = _Sentinel()

        def _reject(cls, *a, **kw):
            raise ValueError("Unrecognized configuration class ProGenConfig for AutoModel")

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", classmethod(_reject))
        lm = _LM()
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: lm),
        )

        assert load_hf_model("fake/repo", encoder_only=False) is lm.transformer

    def test_encoder_only_none_infers_from_config(self, monkeypatch):
        import transformers

        text_encoding = _Sentinel()
        monkeypatch.setattr(
            transformers.AutoModelForTextEncoding,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: text_encoding),
        )

        class _Config:
            is_encoder_decoder = True

        assert load_hf_model("fake/repo", encoder_only=None, config=_Config()) is text_encoding

    def test_dtype_none_pins_float32_rather_than_deferring_to_transformers(self, monkeypatch):
        """Regression test: transformers >= 5 treats a bare ``torch_dtype=None`` as
        ``dtype="auto"`` (the checkpoint's own dtype), not the fp32 that
        transformers 4.x defaulted to. ``load_hf_model`` must pin fp32 itself so a
        reload's precision doesn't depend on the installed transformers major.
        """
        import transformers

        seen_kw = {}

        def _capture(cls, *a, **kw):
            seen_kw.update(kw)
            return _Sentinel()

        monkeypatch.setattr(
            transformers.AutoModelForTextEncoding, "from_pretrained", classmethod(_capture)
        )
        load_hf_model("fake/repo", encoder_only=True, dtype=None)
        assert seen_kw["torch_dtype"] == torch.float32


class TestLoadConfigRegistry:
    """`_load_config` rescues architectures registered by a third-party
    package (see `_REGISTRY_PACKAGES`) that AutoConfig doesn't recognize on
    its own -- e.g. multimolecule/proteinbert, which ships no `auto_map` so
    `--trust-remote-code` can't reach it."""

    def test_happy_path_never_consults_the_registry(self, monkeypatch):
        import transformers

        sentinel = _Sentinel()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: sentinel),
        )

        def _unexpected(*a, **kw):
            raise AssertionError("get_config_dict should not run when AutoConfig succeeds")

        monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", _unexpected)

        assert _load_config("fake/repo") is sentinel

    def test_unknown_type_in_registry_imports_module_and_retries(self, monkeypatch):
        import sys
        import types

        import transformers

        monkeypatch.setitem(_REGISTRY_PACKAGES, "made_up_arch", ("made_up_pkg", "made-up-extra"))
        monkeypatch.setitem(sys.modules, "made_up_pkg", types.ModuleType("made_up_pkg"))

        calls = {"n": 0}

        def _from_pretrained(cls, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("Unrecognized model type made_up_arch")
            return _Sentinel()

        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", classmethod(_from_pretrained)
        )
        monkeypatch.setattr(
            transformers.PretrainedConfig,
            "get_config_dict",
            classmethod(lambda cls, *a, **kw: ({"model_type": "made_up_arch"}, {})),
        )

        result = _load_config("fake/repo")
        assert isinstance(result, _Sentinel)
        assert calls["n"] == 2

    def test_unknown_type_in_registry_but_package_missing_raises_actionable_error(
        self, monkeypatch
    ):
        import transformers

        monkeypatch.setitem(_REGISTRY_PACKAGES, "made_up_arch", ("made_up_pkg", "made-up-extra"))

        def _reject(cls, *a, **kw):
            raise ValueError("Unrecognized model type made_up_arch")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_reject))
        monkeypatch.setattr(
            transformers.PretrainedConfig,
            "get_config_dict",
            classmethod(lambda cls, *a, **kw: ({"model_type": "made_up_arch"}, {})),
        )

        with pytest.raises(RuntimeError, match=r"pip install plmsommelier\[made-up-extra\]"):
            _load_config("fake/repo")

    def test_unknown_type_not_in_registry_reraises_original_error(self, monkeypatch):
        import transformers

        def _reject(cls, *a, **kw):
            raise ValueError("Unrecognized model type totally_unknown_arch")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_reject))
        monkeypatch.setattr(
            transformers.PretrainedConfig,
            "get_config_dict",
            classmethod(lambda cls, *a, **kw: ({"model_type": "totally_unknown_arch"}, {})),
        )

        with pytest.raises(ValueError, match="totally_unknown_arch"):
            _load_config("fake/repo")


class TestCudaIsUsable:
    """Regression coverage: torch.cuda.is_available() alone doesn't mean the
    installed build has kernels for the visible device (e.g. an sm_61 card
    against a cu13 wheel compiled for sm_75+), which used to crash the first
    kernel launch with 'no kernel image is available for execution on the
    device' instead of falling back to CPU."""

    def _patch(self, monkeypatch, *, available, cc=None, arch_list=None, name="GPU0"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        if cc is not None:
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: cc)
        if arch_list is not None:
            monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name, raising=False)

    def test_no_cuda_device(self, monkeypatch):
        self._patch(monkeypatch, available=False)
        assert _cuda_is_usable() is False

    def test_unsupported_capability_falls_back_with_warning(self, monkeypatch):
        # MX250 (sm_61) against a build compiled for sm_75+.
        self._patch(
            monkeypatch,
            available=True,
            cc=(6, 1),
            arch_list=["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        )
        with pytest.warns(RuntimeWarning, match="not supported by this torch build"):
            assert _cuda_is_usable() is False

    def test_exact_arch_match_is_usable(self, monkeypatch):
        self._patch(monkeypatch, available=True, cc=(8, 6), arch_list=["sm_75", "sm_86"])
        assert _cuda_is_usable() is True

    def test_newer_minor_same_major_is_usable(self, monkeypatch):
        # sm_89 device, build only shipped sm_86 kernels for that major.
        self._patch(monkeypatch, available=True, cc=(8, 9), arch_list=["sm_75", "sm_86"])
        assert _cuda_is_usable() is True

    def test_get_arch_list_raising_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (6, 1))
        monkeypatch.setattr(torch.cuda, "get_arch_list", _boom)
        assert _cuda_is_usable() is False

    def test_empty_arch_list_trusts_is_available(self, monkeypatch):
        # e.g. ROCm builds, which don't report an sm_* arch list.
        self._patch(monkeypatch, available=True, cc=(9, 0), arch_list=[])
        assert _cuda_is_usable() is True


class TestSelectDevice:
    """cuda > mps > cpu, and only when each is actually usable."""

    def _patch(self, monkeypatch, *, cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        if cuda:
            # An empty arch list makes _cuda_is_usable trust is_available(),
            # so these tests can stay about the cuda/mps/cpu ladder itself.
            # get_device_capability still has to resolve, though -- it's
            # called before the (empty) arch list is even checked.
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (8, 0))
            monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    def test_prefers_cuda(self, monkeypatch):
        self._patch(monkeypatch, cuda=True, mps=True)
        assert _select_device().type == "cuda"

    def test_falls_back_to_mps_without_cuda(self, monkeypatch):
        self._patch(monkeypatch, cuda=False, mps=True)
        assert _select_device().type == "mps"

    def test_falls_back_to_cpu_without_cuda_or_mps(self, monkeypatch):
        self._patch(monkeypatch, cuda=False, mps=False)
        assert _select_device().type == "cpu"

    def test_falls_back_to_mps_when_cuda_has_no_usable_kernels(self, monkeypatch):
        """The MX250 case: a visible but unsupported CUDA device must not win
        out over a usable mps device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (6, 1))
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75"])
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "GPU0", raising=False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        with pytest.warns(RuntimeWarning, match="not supported by this torch build"):
            assert _select_device().type == "mps"

    def test_tolerates_a_torch_build_with_no_mps_backend(self, monkeypatch):
        """ROCm and some source builds don't expose ``torch.backends.mps`` at all.

        ``torch.backends`` is itself wrapped in a ``GenericModule`` proxy whose
        ``__getattr__`` falls back to the real underlying module (``.m``), so
        deleting ``mps`` from the proxy alone leaves it reachable through that
        fallback -- both have to be cleared to actually simulate its absence.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delattr(torch.backends, "mps", raising=False)
        monkeypatch.delattr(getattr(torch.backends, "m", torch.backends), "mps", raising=False)
        assert _select_device().type == "cpu"


class TestResolveDtype:
    def test_bfloat16_downgrades_to_float32_on_mps(self):
        with pytest.warns(UserWarning, match="bfloat16 is not supported on mps"):
            dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("mps"), "prott5")
        assert dtype == torch.float32

    def test_bfloat16_is_unaffected_on_cuda_that_supports_it(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "prott5")
        assert dtype == torch.bfloat16

    def test_bfloat16_downgrades_to_float32_on_cuda_without_support(self, monkeypatch):
        """A device (or build) where torch.cuda.is_bf16_supported() itself
        says no -- e.g. an old CUDA toolkit with no bf16 emulation path."""
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        with pytest.warns(UserWarning, match="bfloat16 is not supported on cuda"):
            dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "prott5")
        assert dtype == torch.float32

    def test_is_bf16_supported_raising_downgrades(self, monkeypatch):
        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", _boom)
        with pytest.warns(UserWarning, match="bfloat16 is not supported"):
            dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "m")
        assert dtype == torch.float32

    def test_default_is_fp16_on_cuda_fp32_elsewhere(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert _resolve_dtype(None, {}, torch.device("cuda"), "m") == torch.float16
        assert _resolve_dtype(None, {}, torch.device("cpu"), "m") == torch.float32
        assert _resolve_dtype(None, {}, torch.device("mps"), "m") == torch.float32

    def test_explicit_dtype_wins_over_quirk(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        got = _resolve_dtype(torch.float32, {"dtype": torch.bfloat16}, torch.device("cuda"), "m")
        assert got == torch.float32

    def test_emulated_bf16_downgrades_to_float32(self, monkeypatch):
        """The Turing (sm_75) shape from BUGS.md: torch >= 2.6's
        ``is_bf16_supported()`` with no argument (or ``including_emulation=True``)
        reports True via software emulation even with no bf16 tensor cores.
        ``_bf16_is_usable`` must ask for ``including_emulation=False``
        explicitly, or a checkpoint like ProtT5 silently loads in "bf16" with
        no actual speedup."""

        def _fake(including_emulation=True):
            return including_emulation

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", _fake)
        with pytest.warns(UserWarning, match="bfloat16 is not supported on cuda"):
            dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "prott5")
        assert dtype == torch.float32

    def test_real_hardware_support_is_unaffected(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
        dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "prott5")
        assert dtype == torch.bfloat16

    def test_older_torch_without_the_kwarg_falls_back_to_the_plain_call(self, monkeypatch):
        """A torch build old enough that ``is_bf16_supported`` takes no
        arguments at all must not be treated as unsupported."""

        def _fake():
            return True

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", _fake)
        dtype = _resolve_dtype(None, {"dtype": torch.bfloat16}, torch.device("cuda"), "prott5")
        assert dtype == torch.bfloat16


from types import SimpleNamespace  # noqa: E402

from plmsommelier.model import (  # noqa: E402
    _QUIRKS,
    PLM,
    _causal_mask_limit,
    _length_limit,
    _preprocess_text,
    _tokenization_is_degenerate,
)


class TestQuirks:
    """`_QUIRKS` is the only place a new model family's pretraining
    convention should live -- see CONTRIBUTING.md. bert/albert (ProtBert/
    ProtAlbert) must space-join residues and map rare ones to X exactly like
    T5, or they collapse every sequence to a single [UNK] (BUG_REPORT.md)."""

    @pytest.mark.parametrize("family", ["bert", "albert", "t5"])
    def test_prottrans_families_space_join_and_map_rare_residues(self, family):
        text = _preprocess_text("MSTUZOB", _QUIRKS[family])
        assert text == "M S T X X X X"

    def test_esm_has_no_quirks(self):
        assert "esm" not in _QUIRKS

    def test_bert_without_quirks_is_unspaced(self):
        """Pins the bug itself: with no quirks entry, a raw unspaced sequence
        is what would have reached the tokenizer."""
        assert _preprocess_text("MSTNPKPQR", {}) == "MSTNPKPQR"

    def test_prostt5_prefix_composes_with_the_t5_quirk(self):
        quirks = dict(_QUIRKS["t5"], prefix_text="<AA2fold> ")
        assert _preprocess_text("MST", quirks) == "<AA2fold> M S T"


class TestLengthLimit:
    """Only an *absolute*-position-embedding architecture may be capped by
    `config.max_position_embeddings` -- rotary (ESM-2) and T5 (whose config
    reports a `max_position_embeddings` derived from `n_positions` despite
    having no real positional ceiling) must not be, or every ProtT5 input
    would be silently capped at 512 residues."""

    def test_esm1b_like_absolute_embeddings_are_capped_and_offset(self):
        tok = SimpleNamespace(model_max_length=1000000000000)
        cfg = SimpleNamespace(
            position_embedding_type="absolute",
            max_position_embeddings=1026,
            pad_token_id=1,
        )
        assert _length_limit(tok, cfg) == 1024

    def test_esm2_like_rotary_is_not_capped_by_config(self):
        tok = SimpleNamespace(model_max_length=1000000000000)
        cfg = SimpleNamespace(position_embedding_type="rotary", max_position_embeddings=1026)
        assert _length_limit(tok, cfg) is None

    def test_t5_like_config_is_not_capped_despite_reporting_a_limit(self):
        tok = SimpleNamespace(model_max_length=1000000000000)
        cfg = SimpleNamespace(max_position_embeddings=512)  # no position_embedding_type
        assert _length_limit(tok, cfg) is None

    def test_unbounded_tokenizer_sentinel_is_ignored(self):
        tok = SimpleNamespace(model_max_length=int(1e30))
        cfg = SimpleNamespace()
        assert _length_limit(tok, cfg) is None

    def test_tokenizer_limit_alone_is_used(self):
        tok = SimpleNamespace(model_max_length=1024)
        cfg = SimpleNamespace()
        assert _length_limit(tok, cfg) == 1024

    def test_takes_the_min_of_both(self):
        tok = SimpleNamespace(model_max_length=2048)
        cfg = SimpleNamespace(
            position_embedding_type="absolute", max_position_embeddings=1026, pad_token_id=1
        )
        assert _length_limit(tok, cfg) == 1024

    def test_missing_model_max_length_is_tolerated(self):
        tok = SimpleNamespace()
        cfg = SimpleNamespace()
        assert _length_limit(tok, cfg) is None


class TestCausalMaskLimit:
    """ProGen2 and ProtGPT2 report no usable limit from either
    `tokenizer.model_max_length` or `config.max_position_embeddings`, yet
    every attention block still registers a fixed-size boolean causal-mask
    buffer -- feeding it a longer input crashes inside the model with a raw
    shape mismatch instead of a legible truncation warning. `_causal_mask_limit`
    finds that ceiling empirically instead of hardcoding it per architecture."""

    def test_square_bool_buffer_caps_at_its_size(self):
        m = torch.nn.Module()
        m.register_buffer("mask", torch.zeros(1, 1, 1024, 1024, dtype=torch.bool))
        assert _causal_mask_limit(m) == 1024

    def test_non_bool_buffer_is_ignored(self):
        """ESM-2's `position_ids` buffer is int64 and must not be mistaken
        for a context-window ceiling."""
        m = torch.nn.Module()
        m.register_buffer("position_ids", torch.zeros(1, 1026, dtype=torch.long))
        assert _causal_mask_limit(m) is None

    def test_non_square_bool_buffer_is_ignored(self):
        m = torch.nn.Module()
        m.register_buffer("mask", torch.zeros(4, 8, dtype=torch.bool))
        assert _causal_mask_limit(m) is None

    def test_single_element_square_buffer_is_ignored(self):
        m = torch.nn.Module()
        m.register_buffer("scalar_like", torch.zeros(1, 1, dtype=torch.bool))
        assert _causal_mask_limit(m) is None

    def test_takes_the_minimum_across_nested_modules(self):
        m = torch.nn.Module()
        m.register_buffer("a", torch.zeros(1, 1, 1024, 1024, dtype=torch.bool))
        sub = torch.nn.Module()
        sub.register_buffer("b", torch.zeros(1, 1, 512, 512, dtype=torch.bool))
        m.add_module("sub", sub)
        assert _causal_mask_limit(m) == 512

    def test_no_buffers_returns_none(self):
        assert _causal_mask_limit(torch.nn.Module()) is None


class _StubTokenizer:
    """Mimics the shape `_encode` calls the real HF tokenizer with: a
    padded, optionally truncated batch, returned as tensors."""

    def __init__(self, ids_of: dict[str, list[int]]):
        self.ids_of = ids_of
        self.calls: list[dict] = []

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_special_tokens_mask=False,
        return_tensors=None,
    ):
        self.calls.append(dict(truncation=truncation, max_length=max_length))
        rows = [list(self.ids_of[t]) for t in texts]
        if truncation and max_length is not None:
            rows = [ids[:max_length] for ids in rows]
        if return_tensors is None:
            return {"input_ids": rows}
        width = max(len(ids) for ids in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for i, ids in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = True
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_special_tokens_mask:
            out["special_tokens_mask"] = torch.zeros_like(attention_mask)
        return out


class _FakePLM:
    """Just enough of `PLM` for its unbound `_encode` to run, with no weights."""

    _encode = PLM._encode

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self._max_length = max_length
        self._warned_truncation = False
        self.model_id = "fake/model"
        self.quirks: dict = {}

    def _preprocess(self, seq):
        return _preprocess_text(seq, self.quirks)


class TestEncodeTruncation:
    def test_truncation_and_max_length_are_passed_when_a_limit_exists(self):
        tok = _StubTokenizer({"SHORT": [1, 2, 3]})
        plm = _FakePLM(tok, max_length=1024)
        plm._encode(["SHORT"])
        real_call = tok.calls[-1]
        assert real_call == {"truncation": True, "max_length": 1024}

    def test_no_truncation_kwargs_when_there_is_no_limit(self):
        tok = _StubTokenizer({"SHORT": [1, 2, 3]})
        plm = _FakePLM(tok, max_length=None)
        plm._encode(["SHORT"])
        real_call = tok.calls[-1]
        assert real_call == {"truncation": False, "max_length": None}

    def test_oversized_input_warns_exactly_once(self):
        tok = _StubTokenizer({"LONG": list(range(2000)), "SHORT": [1, 2]})
        plm = _FakePLM(tok, max_length=1024)
        with pytest.warns(RuntimeWarning, match="exceeds this architecture's 1024-token limit"):
            plm._encode(["LONG"])
        assert plm._warned_truncation is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plm._encode(["LONG", "SHORT"])  # already warned -- must not warn again

    def test_input_within_the_limit_never_warns(self):
        tok = _StubTokenizer({"SHORT": [1, 2, 3]})
        plm = _FakePLM(tok, max_length=1024)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plm._encode(["SHORT"])


class TestDegenerateTokenization:
    """The generic guard: catches any future family with the same
    space-join-vs-unspaced mismatch, not just BERT/ALBERT specifically."""

    def test_all_unk_pooled_tokens_is_degenerate(self):
        input_ids = torch.tensor([[101, 100, 102]])  # [CLS] [UNK] [SEP]
        pool_mask = torch.tensor([[False, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=100) is True

    def test_multiple_pooled_tokens_all_unk_is_still_degenerate(self):
        input_ids = torch.tensor([[101, 100, 100, 100, 102]])
        pool_mask = torch.tensor([[False, True, True, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=100) is True

    def test_single_pooled_token_is_degenerate_even_if_not_unk(self):
        input_ids = torch.tensor([[101, 7, 102]])
        pool_mask = torch.tensor([[False, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=100) is True

    def test_normal_per_residue_tokenization_is_not_degenerate(self):
        input_ids = torch.tensor([[101, 5, 6, 7, 8, 102]])
        pool_mask = torch.tensor([[False, True, True, True, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=100) is False

    def test_bpe_merging_several_residues_per_token_is_not_degenerate(self):
        """A BPE vocab (e.g. ProGen2) legitimately merges residues into fewer
        tokens; as long as it's more than one, that's not the bug."""
        input_ids = torch.tensor([[101, 55, 56, 102]])
        pool_mask = torch.tensor([[False, True, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=100) is False

    def test_no_unk_token_id_still_catches_the_single_token_case(self):
        input_ids = torch.tensor([[101, 7, 102]])
        pool_mask = torch.tensor([[False, True, False]])
        assert _tokenization_is_degenerate(input_ids, pool_mask, unk_token_id=None) is True


from plmsommelier.model import (  # noqa: E402
    _is_remote_code,
    _prepare_remote_code_for_save,
    embed_layers,
    load_model,
    locate_blocks,
    save_truncated,
    truncate,
)
from plmsommelier.select import Result  # noqa: E402


class TestTruncateMemory:
    """`truncate()` must deep-copy only the retained blocks, not the whole
    model plus the ones it's about to discard -- a full-size deepcopy briefly
    doubles peak GPU memory on top of whatever the just-finished embedding
    pass left reserved (BUGS.md)."""

    def _stub_model(self, n: int = 4) -> torch.nn.Module:
        m = torch.nn.Module()
        m.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(n)])
        m.config = SimpleNamespace(num_hidden_layers=n)
        return m

    def test_deepcopy_sees_only_the_retained_blocks(self, monkeypatch):
        model = self._stub_model(4)
        plm_stub = SimpleNamespace(model=model, n_layers=4)
        seen_lengths = []
        real_deepcopy = copy.deepcopy

        def _spy(obj, *a, **kw):
            _, _, blocks = locate_blocks(obj)
            seen_lengths.append(len(blocks))
            return real_deepcopy(obj, *a, **kw)

        monkeypatch.setattr(copy, "deepcopy", _spy)
        truncated = truncate(plm_stub, 2)

        assert seen_lengths == [2]  # never the full 4
        assert len(truncated.layers) == 2
        assert len(model.layers) == 4  # the original is restored, not left shrunk

    def test_original_block_list_is_restored_even_if_deepcopy_raises(self, monkeypatch):
        model = self._stub_model(4)
        plm_stub = SimpleNamespace(model=model, n_layers=4)

        def _boom(obj, *a, **kw):
            raise RuntimeError("simulated OOM")

        monkeypatch.setattr(copy, "deepcopy", _boom)
        with pytest.raises(RuntimeError, match="simulated OOM"):
            truncate(plm_stub, 2)
        assert len(model.layers) == 4


def _remote_code_class(module_name: str, class_name: str = "Stub") -> type:
    """A class whose `__module__` looks like one transformers dynamically
    imported via `trust_remote_code` (see `get_class_from_dynamic_module`)."""
    return type(class_name, (), {"__module__": module_name})


class _FakeConfig:
    def __init__(self, auto_map: dict[str, str]):
        self.auto_map = auto_map


class TestPrepareRemoteCodeForSave:
    """Regression coverage for the ProGen2 "config_class attribute is not
    consistent" reload failure: a stale hub-prefixed `auto_map` entry
    surviving a save must not linger once ``register_for_auto_class`` has
    made the config self-referential -- see CONTRIBUTING.md invariant 7."""

    def test_non_remote_code_model_is_untouched(self):
        model = _remote_code_class("transformers.models.esm.modeling_esm", "EsmModel")()
        model.config = _FakeConfig({"AutoModel": "esm/repo--modeling_esm.EsmModel"})
        assert _is_remote_code(model) is False

        _prepare_remote_code_for_save(model, "AutoModel")

        assert not hasattr(model, "_auto_class")
        assert not hasattr(model.config, "_auto_class")
        assert model.config.auto_map == {"AutoModel": "esm/repo--modeling_esm.EsmModel"}

    def test_remote_code_model_gets_a_clean_self_consistent_auto_map(self):
        stale_auto_map = {
            "AutoConfig": "hugohrban/progen2-small--configuration_progen.ProGenConfig",
            "AutoModelForCausalLM": "hugohrban/progen2-small--modeling_progen.ProGenForCausalLM",
        }
        Model = _remote_code_class(
            "transformers_modules.hugohrban.progen2-small.abc123.modeling_progen", "ProGenModel"
        )
        model = Model()
        model.config = _FakeConfig(dict(stale_auto_map))
        assert _is_remote_code(model) is True

        _prepare_remote_code_for_save(model, "AutoModel")

        # Every stale hub-prefixed entry is gone -- including one for a class
        # (the causal-LM head) truncation removed entirely -- so
        # save_pretrained's own custom_object_save starts from a clean slate
        # and fills in only entries that resolve to files it actually copies.
        assert model.config.auto_map == {}
        assert model._auto_class == "AutoModel"
        assert model.config._auto_class == "AutoConfig"

    def test_encoder_only_auto_class_is_honored(self):
        Model = _remote_code_class("transformers_modules.foo.bar.modeling_foo", "FooModel")
        model = Model()
        model.config = _FakeConfig({})
        _prepare_remote_code_for_save(model, "AutoModelForTextEncoding")
        assert model._auto_class == "AutoModelForTextEncoding"

    def test_never_mutates_the_class_itself(self):
        """`register_for_auto_class` sets a *class*-level flag that leaks
        into every future save of that architecture in the same process --
        exactly how the bug's stale entries kept reappearing. Only instance
        attributes may be touched."""
        Model = _remote_code_class("transformers_modules.foo.bar.modeling_foo", "FooModel")
        model = Model()
        model.config = _FakeConfig({})
        _prepare_remote_code_for_save(model, "AutoModel")
        assert "_auto_class" not in vars(Model)


@pytest.fixture(scope="module")
def plm():
    return load_model("facebook/esm2_t6_8M_UR50D")


def _result(layer: int) -> Result:
    return Result(
        model="facebook/esm2_t6_8M_UR50D",
        dataset="synthetic",
        task="regression",
        probe="knn",
        best_layer=layer,
        best_score=0.60,
        last_layer=6,
        last_layer_score=0.50,
        curve={layer: 0.60, 6: 0.50},
        plateau=[layer],
        n_train=100,
        n_val=50,
        seed_layers=[layer] * 3,
        seed_agreement=1.0,
    )


@pytest.mark.weights
class TestEmbedLayers:
    def test_shape_and_dtype(self, plm):
        out = embed_layers(plm, ["ACDEFGHIKLMNPQRSTVWY", "MKT"])
        assert out.shape == (plm.n_states, 2, plm.hidden_size)
        assert out.dtype == np.float16


@pytest.mark.weights
class TestTruncationRoundTrip:
    """The deliverable must survive a round-trip through from_pretrained.

    ESM's contact_head is Linear(n_layers * n_heads, 1), so a truncated model
    whose head was left at full width writes a checkpoint that no longer
    matches its own config -- which only surfaces on reload.
    """

    @pytest.mark.parametrize("layer", [0, 1, 2, 4, 6])
    def test_saved_model_reloads_and_reproduces_the_layer(self, plm, tmp_path, layer):
        from transformers import AutoModel

        out = save_truncated(plm, _result(layer), tmp_path / f"l{layer}")

        # Reload in the same dtype the fixture ran in. On CUDA that's fp16
        # (see _resolve_dtype); an unpinned reload upcasts to fp32 under
        # transformers 4.x, and the fp16 rounding noise (~4e-3) swamps the
        # tolerance below even though nothing is actually wrong.
        reloaded = (
            AutoModel.from_pretrained(out, add_pooling_layer=False, torch_dtype=plm.model.dtype)
            .to(plm.device)
            .eval()
        )
        assert reloaded.config.num_hidden_layers == layer

        states, _ = plm.forward(["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"])
        reference = states[layer]

        batch_inputs, _ = plm._encode(["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"])
        with torch.no_grad():
            got = reloaded(**batch_inputs).last_hidden_state
        diff = float((got.float() - reference.float()).abs().max())
        assert diff <= 1e-4, f"reloaded model drifted: {diff:.3e}"

    def test_contact_head_matches_the_retained_depth(self, plm):
        n_heads = plm.model.config.num_attention_heads
        for layer in (0, 2, 6):
            head = truncate(plm, layer).contact_head.regression
            assert head.in_features == layer * n_heads

    def test_contact_head_keeps_the_model_dtype(self):
        """Regression test: the resized `nn.Linear` used to be built with
        `nn.Linear`'s own fp32/CPU default instead of matching the original
        head's dtype/device, so a model loaded in fp16 got a mixed-precision
        (and, on CUDA, mixed-device) checkpoint after truncation."""
        fp16_plm = load_model("facebook/esm2_t6_8M_UR50D", device="cpu", dtype=torch.float16)
        head = truncate(fp16_plm, 2).contact_head.regression
        assert head.weight.dtype == torch.float16
        assert head.weight.device == fp16_plm.model.device

    def test_provenance_is_recorded(self, plm, tmp_path):
        out = save_truncated(plm, _result(2), tmp_path / "prov")
        meta = json.loads((out / "plmsommelier.json").read_text())
        assert meta["base_model"] == "facebook/esm2_t6_8M_UR50D"
        assert meta["layers_kept"] == 2 and meta["layers_total"] == 6

        card = (out / "README.md").read_text()
        assert "truncated to 2 of 6 layers" in card
        assert "plmsommelier" in card and meta["dataset"] in card

    def test_truncated_model_is_smaller(self, plm, tmp_path):
        from transformers import AutoModel

        small = AutoModel.from_pretrained(
            save_truncated(plm, _result(2), tmp_path / "s"), add_pooling_layer=False
        )
        n_small = sum(p.numel() for p in small.parameters())
        n_full = sum(p.numel() for p in plm.model.parameters())
        assert n_small < n_full


@pytest.fixture(scope="module")
def progen2_plm():
    return load_model("hugohrban/progen2-small", trust_remote_code=True)


def _progen2_result(plm, layer: int) -> Result:
    return Result(
        model=plm.model_id,
        dataset="synthetic",
        task="regression",
        probe="knn",
        best_layer=layer,
        best_score=0.60,
        last_layer=plm.n_layers,
        last_layer_score=0.50,
        curve={layer: 0.60, plm.n_layers: 0.50},
        plateau=[layer],
        n_train=10,
        n_val=5,
        seed_layers=[layer] * 3,
        seed_agreement=1.0,
    )


@pytest.mark.weights
class TestProGen2CausalMaskLimit:
    """ProGen2's 1024-token context window is enforced only by a fixed-size
    causal-mask buffer, not by anything `tokenizer.model_max_length` or
    `config.max_position_embeddings` reports -- see `_causal_mask_limit`."""

    def test_max_length_is_detected_from_the_causal_mask(self, progen2_plm):
        assert progen2_plm._max_length == 1024

    def test_an_oversized_sequence_warns_and_embeds_instead_of_crashing(self, progen2_plm):
        with pytest.warns(RuntimeWarning, match="1024-token limit"):
            out = embed_layers(progen2_plm, ["A" * 1500])
        assert out.shape == (progen2_plm.n_states, 1, progen2_plm.hidden_size)


@pytest.mark.weights
class TestRemoteCodeTruncationRoundTrip:
    """Regression coverage for the exact benchmark failure in BUGS.md: every
    ProGen2 (model, dataset) pair after the first failed to reload, because
    the first local reload permanently registers the config class for
    self-referential auto-loading, and a second, differently-named save
    directory still inherits the base checkpoint's stale hub-prefixed
    `AutoModelForCausalLM` entry."""

    def test_two_separate_saves_of_the_same_model_both_reload(self, progen2_plm, tmp_path):
        from plmsommelier.model import load_hf_model

        result = _progen2_result(progen2_plm, layer=2)
        out_a = save_truncated(progen2_plm, result, tmp_path / "a")
        out_b = save_truncated(progen2_plm, result, tmp_path / "b")

        reference, _ = progen2_plm.forward(["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"])
        model_dtype = next(progen2_plm.model.parameters()).dtype

        for out in (out_a, out_b):
            reloaded = (
                load_hf_model(out, encoder_only=False, trust_remote_code=True, dtype=model_dtype)
                .to(progen2_plm.device)
                .eval()
            )
            assert reloaded.config.n_layer == 2

            batch_inputs, _ = progen2_plm._encode(
                ["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"]
            )
            with torch.no_grad():
                got = reloaded(**batch_inputs).last_hidden_state
            # A looser tolerance than the ESM round-trip test: ProGen2's rotary
            # attention accumulates more fp16 rounding noise across a forward
            # pass than ESM's absolute embeddings do, even reloaded in the
            # same dtype the fixture ran in (values here run up to ~50).
            diff = float((got.float() - reference[2].float()).abs().max())
            assert diff <= 5e-2, f"reloaded model drifted: {diff:.3e}"


try:
    import multimolecule  # noqa: F401

    _MULTIMOLECULE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent
    # A bare ImportError means the optional extra isn't installed; anything
    # else (e.g. a transformers version whose own architectures collide with
    # ones multimolecule registers under the same name) is a real, separate
    # incompatibility -- either way, skip rather than fail collection, since
    # `pytest.importorskip` only catches ImportError and this can raise
    # ValueError instead.
    _MULTIMOLECULE_IMPORT_ERROR = exc


@pytest.fixture(scope="module")
def proteinbert_plm():
    return load_model("multimolecule/proteinbert")


@pytest.mark.weights
@pytest.mark.skipif(
    _MULTIMOLECULE_IMPORT_ERROR is not None,
    reason=f"multimolecule unavailable: {_MULTIMOLECULE_IMPORT_ERROR}",
)
class TestMultimoleculeRegistry:
    """End-to-end coverage for the third-party registry hook: proteinbert's
    architecture is registered by the multimolecule package rather than by
    transformers, and its config carries no `auto_map`, so this is the one
    family --trust-remote-code cannot rescue."""

    def test_loads_and_reports_every_hidden_state(self, proteinbert_plm):
        assert proteinbert_plm.n_states == proteinbert_plm.config.num_hidden_layers + 1

    def test_embed_layers_shape(self, proteinbert_plm):
        out = embed_layers(proteinbert_plm, ["ACDEFGHIKLMNPQRSTVWY", "MKT"])
        assert out.shape == (proteinbert_plm.n_states, 2, proteinbert_plm.hidden_size)
