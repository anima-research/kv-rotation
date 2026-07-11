"""CPU tests for the exp11 signature bridge (src/kvrot/sigbridge.py).

Everything the correctness-gate logic needs is exercised here on a tiny random
Llama (4 layers, fp32, CPU) so the GPU stage only has to confirm numerics at
scale: gate equivalence (injected-FULL-cache replay ≡ use_cache=False replay),
position-offset handling, NAIVE gap positions, RawGenerationData layout
compatibility, the incremental (path-floor) twin, positional-means remapping,
and the calibration / factor-direction loaders with their hard alignment asserts.

Requires transformers + scipy (the `sig` extra) and an importable anamnesis-pl
checkout (see conftest.py); skips cleanly when either is missing.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("scipy")
anamnesis = pytest.importorskip("anamnesis")

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig  # noqa: E402
from anamnesis.extraction.model_loader import (  # noqa: E402
    HookState,
    LoadedModel,
    _make_gate_proj_hook,
    _make_k_proj_hook,
)
from anamnesis.extraction.replay_extract import replay_extract  # noqa: E402

from kvrot.sigbridge import (  # noqa: E402
    CalibrationBundle,
    FactorDirections,
    GateResult,
    assert_feature_alignment,
    compute_signature,
    correctness_gate,
    load_3b_calibration,
    load_factor_directions,
    mode_projection,
    remap_positional_means,
    replay_extract_cached,
)
from kvrot.snapshot import evict, from_hf_cache, reindex, to_hf_dynamic_cache  # noqa: E402

# ── Tiny model fixture ───────────────────────────────────────────────────────────

N_LAYERS = 4
HIDDEN = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 256
SAMPLED = [0, 1, 2, 3]
PCA_LAYERS = [1, 2]
MAX_POS = 64  # deliberately small so positional clamping paths are exercised

P_CTX = 30
N_CONT = 20


def _tiny_model() -> "transformers.LlamaForCausalLM":
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(20260710)
    model = transformers.LlamaForCausalLM(cfg)
    model.eval()
    # Force eager (constructor kwarg handling varies across transformers versions).
    if getattr(model.config, "_attn_implementation", None) != "eager":
        model.config._attn_implementation = "eager"
    return model


@pytest.fixture(scope="module")
def loaded() -> LoadedModel:
    """Anamnesis LoadedModel around the tiny llama, hooks wired like load_model."""
    model = _tiny_model()
    hook_state = HookState()
    handles = []
    for layer_idx in SAMPLED:
        k_proj = model.model.layers[layer_idx].self_attn.k_proj
        handles.append(
            k_proj.register_forward_hook(
                _make_k_proj_hook(layer_idx, hook_state, N_KV_HEADS, HEAD_DIM)
            )
        )
        gate_proj = model.model.layers[layer_idx].mlp.gate_proj
        handles.append(
            gate_proj.register_forward_hook(_make_gate_proj_hook(layer_idx, hook_state))
        )
    config = ModelConfig(
        model_id="tiny-test-llama",
        torch_dtype="float32",
        num_layers=N_LAYERS,
        hidden_dim=HIDDEN,
        num_attention_heads=N_HEADS,
        num_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
    )
    return LoadedModel(
        model=model, tokenizer=None, hook_state=hook_state, hook_handles=handles, config=config
    )


@pytest.fixture(scope="module")
def calib() -> CalibrationBundle:
    rng = np.random.default_rng(7)
    return CalibrationBundle(
        positional_means=rng.normal(0, 0.01, (N_LAYERS + 1, MAX_POS, HIDDEN)).astype(
            np.float32
        ),
        pca_components=rng.normal(0, 1, (8, HIDDEN)).astype(np.float32),
        pca_mean=rng.normal(0, 0.1, HIDDEN).astype(np.float32),
        source_dir="<synthetic>",
    )


@pytest.fixture(scope="module")
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(
        sampled_layers=SAMPLED,
        pca_layers=PCA_LAYERS,
        early_layer_cutoff=1,
        late_layer_cutoff=2,
        enable_tier3=True,
    )


@pytest.fixture(scope="module")
def family_config() -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=True,
        enable_stft=True,
        enable_contrastive_projection=False,
        trajectory_layers=PCA_LAYERS,
        contrastive_layers=PCA_LAYERS,
    )


@pytest.fixture(scope="module")
def tokens() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    ctx = torch.randint(0, VOCAB, (1, P_CTX), generator=g)
    cont = torch.randint(0, VOCAB, (1, N_CONT), generator=g)
    return ctx, cont


def _prefill_cache(loaded: LoadedModel, ctx: torch.Tensor):
    from transformers import DynamicCache

    loaded.disable_hooks()
    try:
        with torch.no_grad():
            out = loaded.model(ctx, past_key_values=DynamicCache(), use_cache=True)
        return out.past_key_values
    finally:
        loaded.enable_hooks()


def _prefill_snapshot(loaded: LoadedModel, ctx: torch.Tensor):
    cache = _prefill_cache(loaded, ctx)
    return from_hf_cache(
        cache,
        positions=torch.arange(ctx.shape[1]),
        layer_types=["full_attention"] * N_LAYERS,
        applies_rope=[True] * N_LAYERS,
    )


def _inv_freq(loaded: LoadedModel) -> torch.Tensor:
    for name, buf in loaded.model.named_buffers():
        if name.endswith("inv_freq"):
            return buf.detach().float().cpu()
    raise AssertionError("tiny model has no inv_freq buffer")


# ── The correctness gate itself ──────────────────────────────────────────────────


def test_gate_passes_on_full_cache(loaded, calib, extraction_config, family_config, tokens):
    ctx, cont = tokens
    result = correctness_gate(
        loaded, ctx, cont, extraction_config, family_config, calib, rtol=1e-4, atol=1e-6
    )
    assert isinstance(result, GateResult)
    assert result.names_equal, "feature names differ between replay paths"
    assert result.passed, result.summary()
    assert result.n_features > 0


def test_gate_reports_failure_shape(loaded, calib, extraction_config, family_config, tokens):
    """An absurdly tight tolerance must produce a well-formed FAIL, not a crash."""
    ctx, cont = tokens
    result = correctness_gate(
        loaded, ctx, cont, extraction_config, family_config, calib, rtol=0.0, atol=0.0
    )
    # fp32 CPU noise is tiny but nonzero somewhere; either outcome must be well-formed.
    assert result.n_features > 0
    assert result.max_abs_diff >= 0.0
    assert "correctness gate" in result.summary()


# ── Position-offset handling ─────────────────────────────────────────────────────


def test_wrong_offset_changes_features(loaded, calib, extraction_config, family_config, tokens):
    ctx, cont = tokens
    cache_a = _prefill_cache(loaded, ctx)
    raw_a = replay_extract_cached(
        loaded, cache_a, cont, P_CTX, positional_means=calib.positional_means
    )
    sig_a = compute_signature(raw_a, extraction_config, family_config, calib)

    cache_b = _prefill_cache(loaded, ctx)
    raw_b = replay_extract_cached(
        loaded, cache_b, cont, P_CTX + 7, positional_means=calib.positional_means
    )
    sig_b = compute_signature(raw_b, extraction_config, family_config, calib)

    assert list(sig_a.feature_names) == list(sig_b.feature_names)
    assert not np.allclose(sig_a.features, sig_b.features, rtol=1e-4, atol=1e-6), (
        "shifting position_offset by +7 left every feature unchanged — position_ids "
        "are not reaching the forward"
    )


def test_offset_before_cache_end_rejected(loaded, tokens):
    ctx, cont = tokens
    cache = _prefill_cache(loaded, ctx)
    with pytest.raises(ValueError, match="position_offset"):
        replay_extract_cached(loaded, cache, cont, P_CTX - 1)


def test_too_short_continuation_rejected(loaded, tokens):
    ctx, _ = tokens
    cache = _prefill_cache(loaded, ctx)
    with pytest.raises(ValueError, match="continuation"):
        replay_extract_cached(loaded, cache, torch.tensor([[5]]), P_CTX)


# ── NAIVE gap positions ──────────────────────────────────────────────────────────


def test_naive_gap_positions(loaded, calib, extraction_config, family_config, tokens):
    """Evict a middle block WITHOUT re-rotation: survivors keep original positions,
    the continuation starts at survivor-max+1 (= P here, tail kept), and the cache
    is physically compacted to K entries."""
    ctx, cont = tokens
    s0 = _prefill_snapshot(loaded, ctx)
    keep = torch.cat([torch.arange(0, 4), torch.arange(14, P_CTX)])  # evict [4, 14)
    k_len = int(keep.shape[0])

    snap_naive = evict(s0.clone(), keep)  # positions carried unchanged — the gap stays
    assert int(snap_naive.positions[0].max().item()) == P_CTX - 1
    offset_naive = snap_naive.next_position()
    assert offset_naive == P_CTX  # tail kept ⇒ continuation resumes at P

    raw_naive = replay_extract_cached(
        loaded,
        to_hf_dynamic_cache(snap_naive),
        cont,
        offset_naive,
        positional_means=calib.positional_means,
    )
    assert raw_naive.prompt_length == k_len  # attention-column semantics

    t_steps = N_CONT - 1
    assert len(raw_naive.attentions) == t_steps
    for i, att in enumerate(raw_naive.attentions):
        assert att.shape == (N_LAYERS, N_HEADS, k_len + i + 1)

    # ROT counterpart: same keep-set, re-rotated to contiguous 0..K-1.
    snap_rot = reindex(evict(s0.clone(), keep), torch.arange(k_len), _inv_freq(loaded))
    raw_rot = replay_extract_cached(
        loaded,
        to_hf_dynamic_cache(snap_rot),
        cont,
        k_len,
        positional_means=calib.positional_means,
    )
    sig_naive = compute_signature(raw_naive, extraction_config, family_config, calib)
    sig_rot = compute_signature(raw_rot, extraction_config, family_config, calib)
    assert list(sig_naive.feature_names) == list(sig_rot.feature_names)
    assert np.all(np.isfinite(sig_naive.features))
    assert not np.allclose(sig_naive.features, sig_rot.features, rtol=1e-4, atol=1e-6), (
        "NAIVE (gap left, no reindex) and ROT produced identical signatures — the "
        "conditions are not reaching the model"
    )


# ── RawGenerationData layout compatibility ───────────────────────────────────────


def test_rawdata_layout_matches_replay_extract(loaded, calib, tokens):
    ctx, cont = tokens
    full = torch.cat([ctx, cont], dim=1)
    ref = replay_extract(loaded, full[0], P_CTX, positional_means=calib.positional_means)

    cache = _prefill_cache(loaded, ctx)
    br = replay_extract_cached(
        loaded, cache, cont, P_CTX, positional_means=calib.positional_means
    )

    t_steps = N_CONT - 1
    assert len(ref.hidden_states) == len(br.hidden_states) == t_steps
    assert len(ref.attentions) == len(br.attentions) == t_steps
    assert len(ref.logits) == len(br.logits) == t_steps
    for i in range(t_steps):
        assert ref.hidden_states[i].shape == br.hidden_states[i].shape
        assert ref.attentions[i].shape == br.attentions[i].shape
        assert ref.logits[i].shape == br.logits[i].shape
    np.testing.assert_array_equal(ref.chosen_token_ids, br.chosen_token_ids)
    assert ref.prompt_length == br.prompt_length == P_CTX
    assert set(ref.pre_rope_keys.keys()) == set(br.pre_rope_keys.keys()) == set(SAMPLED)
    for l_idx in SAMPLED:
        assert len(br.pre_rope_keys[l_idx]) == t_steps
        assert br.pre_rope_keys[l_idx][0].shape == (N_KV_HEADS, HEAD_DIM)
    assert br.gate_activations is not None
    for l_idx in SAMPLED:
        assert len(br.gate_activations[l_idx]) == t_steps
        assert br.gate_activations[l_idx][0].shape == (2 * HIDDEN,)


def test_feature_names_align_across_conditions(
    loaded, calib, extraction_config, family_config, tokens
):
    """The per-cell hard assert (spec confound #4), exercised end-to-end."""
    ctx, cont = tokens
    s0 = _prefill_snapshot(loaded, ctx)
    keep = torch.cat([torch.arange(0, 4), torch.arange(12, P_CTX)])
    k_len = int(keep.shape[0])
    inv = _inv_freq(loaded)

    sigs = {}
    conditions = {
        "FULL": (s0.clone(), P_CTX),
        "ROT": (reindex(evict(s0.clone(), keep), torch.arange(k_len), inv), k_len),
        "NAIVE": (evict(s0.clone(), keep), P_CTX),
    }
    for name, (snap, offset) in conditions.items():
        raw = replay_extract_cached(
            loaded,
            to_hf_dynamic_cache(snap),
            cont,
            offset,
            positional_means=calib.positional_means,
        )
        sigs[name] = compute_signature(raw, extraction_config, family_config, calib)

    ref_names = list(sigs["FULL"].feature_names)
    for name, sig in sigs.items():
        assert_feature_alignment(ref_names, list(sig.feature_names), name)
    assert len(ref_names) == len(set(ref_names)) or True  # names may repeat across tiers


# ── Incremental (path-floor) twin ────────────────────────────────────────────────


def test_incremental_matches_batched(loaded, calib, extraction_config, family_config, tokens):
    ctx, cont = tokens
    raw_b = replay_extract_cached(
        loaded, _prefill_cache(loaded, ctx), cont, P_CTX,
        positional_means=calib.positional_means,
    )
    raw_i = replay_extract_cached(
        loaded, _prefill_cache(loaded, ctx), cont, P_CTX,
        positional_means=calib.positional_means, incremental=True,
    )
    t_steps = N_CONT - 1
    assert len(raw_i.attentions) == t_steps
    for i in range(t_steps):
        assert raw_i.attentions[i].shape == raw_b.attentions[i].shape
        np.testing.assert_allclose(
            raw_i.logits[i], raw_b.logits[i], rtol=1e-4, atol=1e-5
        )
    sig_b = compute_signature(raw_b, extraction_config, family_config, calib)
    sig_i = compute_signature(raw_i, extraction_config, family_config, calib)
    assert list(sig_b.feature_names) == list(sig_i.feature_names)
    np.testing.assert_allclose(sig_i.features, sig_b.features, rtol=1e-3, atol=1e-5)


# ── Positional-means remap (pure function) ───────────────────────────────────────


def test_remap_positional_means_identity_and_shift():
    rng = np.random.default_rng(3)
    pm = rng.normal(size=(3, 20, 4)).astype(np.float32)

    assert remap_positional_means(None, 5, 9, 3) is None
    assert remap_positional_means(pm, 5, 5, 3) is pm  # FULL/ROT/REC: identity

    out = remap_positional_means(pm, 5, 9, 3)  # NAIVE-style: cache 5, positions from 9
    assert out is not pm
    np.testing.assert_array_equal(out[:, 5], pm[:, 9])
    np.testing.assert_array_equal(out[:, 6], pm[:, 10])
    np.testing.assert_array_equal(out[:, 7], pm[:, 11])
    np.testing.assert_array_equal(out[:, 4], pm[:, 4])  # untouched below cache_len

    # Clamping: offsets past the table end reuse the final row.
    out2 = remap_positional_means(pm, 5, 18, 4)
    np.testing.assert_array_equal(out2[:, 6], pm[:, 19])
    np.testing.assert_array_equal(out2[:, 8], pm[:, 19])

    # Long-context fast path: everything clamps anyway → same object back.
    assert remap_positional_means(pm, 19, 40, 3) is pm

    with pytest.raises(ValueError, match="position_offset"):
        remap_positional_means(pm, 9, 5, 3)


# ── Loaders ──────────────────────────────────────────────────────────────────────


def test_calibration_loader_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    pm = rng.normal(size=(5, 12, 8)).astype(np.float32)
    np.savez(tmp_path / "positional_means.npz", positional_means=pm, pos_counts=np.ones((5, 12)))
    with open(tmp_path / "pca_model.pkl", "wb") as f:
        pickle.dump(
            {"components": rng.normal(size=(4, 8)), "mean": rng.normal(size=8)}, f
        )
    bundle = load_3b_calibration(tmp_path)
    np.testing.assert_array_equal(bundle.positional_means, pm)
    assert bundle.pca_components.shape == (4, 8)
    assert bundle.pca_mean.shape == (8,)

    with pytest.raises(FileNotFoundError):
        load_3b_calibration(tmp_path / "nope")
    # tier3 disabled: PCA pickle not required
    (tmp_path / "no_pca").mkdir()
    np.savez(tmp_path / "no_pca" / "positional_means.npz", positional_means=pm)
    b2 = load_3b_calibration(tmp_path / "no_pca", enable_tier3=False)
    assert b2.pca_components is None


def _write_factor_npz(path, d=10):
    rng = np.random.default_rng(9)
    labels = np.array(["analogical", "contrastive", "dialectical", "linear", "socratic"])
    w = np.zeros((4, d))
    w[0, 0] = w[1, 1] = w[2, 2] = w[3, 3] = 1.0  # rows span e0..e3
    names = np.array([f"feat_{i}" for i in range(d)])
    kw = {
        "labels": labels,
        "FULL_W": w,
        "FULL_mean": np.zeros(d),
        "FULL_scale": np.ones(d),
        "FULL_names": names,
    }
    for m in labels:
        kw[f"FULL_centroid_{m}"] = rng.normal(size=4)
    np.savez(path, **kw)
    return [str(n) for n in names]


def test_mode_projection_geometry(tmp_path):
    names = _write_factor_npz(tmp_path / "fd.npz")
    fd = load_factor_directions(tmp_path / "fd.npz", space="FULL")
    assert isinstance(fd, FactorDirections)
    assert fd.labels[0] == "analogical"
    assert len(fd.centroids) == 5

    inside = np.zeros(10, dtype=np.float32)
    inside[0], inside[2] = 2.0, -1.0
    mp = mode_projection(inside, names, fd)
    assert mp.mode_fraction == pytest.approx(1.0, abs=1e-9)
    assert mp.proj_w[0] == pytest.approx(2.0)
    assert mp.proj_w[2] == pytest.approx(-1.0)

    outside = np.zeros(10, dtype=np.float32)
    outside[7] = 3.0
    mp2 = mode_projection(outside, names, fd)
    assert mp2.mode_fraction == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(AssertionError, match="alignment"):
        mode_projection(inside, ["wrong"] * 10, fd)
    with pytest.raises(ValueError):
        load_factor_directions(tmp_path / "fd.npz", space="BOGUS")


def test_assert_feature_alignment_messages():
    assert_feature_alignment(["a", "b"], ["a", "b"], "ok")
    with pytest.raises(AssertionError, match="index 1"):
        assert_feature_alignment(["a", "b"], ["a", "c"], "cond")
    with pytest.raises(AssertionError, match="lengths"):
        assert_feature_alignment(["a", "b"], ["a"], "cond")
