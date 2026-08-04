"""Invariants of the FAST pipeline.

These guard the properties the experiments depend on being true: the transform
is invertible, quantization is the only lossy step, and the compression claim
holds in the direction the paper says it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastlab import FastConfig, FastTokenizer, NaiveBinningTokenizer, reconstruction_error
from fastlab import dct as dct_ops
from fastlab import normalize as norm_ops
from fastlab.data import gripper_like_chunks, synthetic_chunks

HORIZON, ACTION_DIM = 32, 7


@pytest.fixture(scope="module")
def chunks() -> np.ndarray:
    return synthetic_chunks(
        n_chunks=400, horizon=HORIZON, action_dim=ACTION_DIM, control_hz=50.0, seed=7
    )


@pytest.fixture(scope="module")
def tokenizer(chunks) -> FastTokenizer:
    return FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=512))


# -- individual stages ---------------------------------------------------


def test_dct_is_invertible(chunks):
    """Without quantization the transform must round-trip to machine precision."""
    c = chunks[0]
    assert np.allclose(dct_ops.inverse_dct(dct_ops.forward_dct(c)), c, atol=1e-10)


def test_normalization_round_trips_without_clipping(chunks):
    stats = norm_ops.fit(chunks)
    normed = norm_ops.apply(chunks[:16], stats, clip=False)
    assert np.allclose(norm_ops.invert(normed, stats), chunks[:16], atol=1e-8)


def test_clipping_is_lossy_only_outside_the_quantile_band(chunks):
    """Quantile normalization trades exactness at the tails for range.

    That is the whole reason FAST recommends quantiles over min/max: a few
    outlier timesteps must not squeeze the rest of the distribution toward zero.
    """
    stats = norm_ops.fit(chunks)
    sample = chunks[:16]
    normed = norm_ops.apply(sample, stats)
    assert normed.min() >= -1.0 and normed.max() <= 1.0

    restored = norm_ops.invert(normed, stats)
    inside = (sample >= stats.low) & (sample <= stats.high)
    assert np.allclose(restored[inside], sample[inside], atol=1e-8)
    assert not np.allclose(restored[~inside], sample[~inside], atol=1e-8)


def test_constant_dimension_does_not_divide_by_zero():
    flat = np.zeros((10, HORIZON, 3))
    flat[..., 1] = 5.0  # a dimension with no variation at all
    stats = norm_ops.fit(flat)
    out = norm_ops.apply(flat, stats)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("order", ["freq_major", "dim_major"])
def test_flatten_round_trips(order):
    m = np.arange(HORIZON * ACTION_DIM).reshape(HORIZON, ACTION_DIM)
    flat = dct_ops.flatten(m, order)
    assert np.array_equal(dct_ops.unflatten(flat, HORIZON, ACTION_DIM, order), m)


def test_unflatten_pads_a_truncated_stream():
    """A short stream is the normal case: trailing zeros are never transmitted."""
    m = dct_ops.unflatten(np.array([1, 2, 3]), HORIZON, ACTION_DIM, "freq_major")
    assert m.shape == (HORIZON, ACTION_DIM)
    assert m.reshape(-1)[3:].sum() == 0


def test_quantize_zeroes_small_coefficients():
    coeffs = np.array([[0.001, 0.04], [0.9, -0.6]])
    q = dct_ops.quantize(coeffs, scale=10.0)
    assert q[0, 0] == 0 and q[0, 1] == 0  # below 1/(2*scale)
    assert q[1, 0] == 9 and q[1, 1] == -6


# -- end to end ----------------------------------------------------------


def test_round_trip_is_close(tokenizer, chunks):
    recon = tokenizer.decode_batch(tokenizer.encode_batch(chunks[:32]))
    assert recon.shape == chunks[:32].shape
    assert reconstruction_error(chunks[:32], recon).normalized_rmse < 0.15


def test_beats_naive_binning(tokenizer, chunks):
    tokens = tokenizer.encode_batch(chunks[:64])
    mean = np.mean([len(t) for t in tokens])
    assert mean < HORIZON * ACTION_DIM, "FAST must emit fewer tokens than T*D"


def test_higher_scale_trades_tokens_for_accuracy(chunks):
    """The core rate-distortion direction, which experiment 03 plots."""
    coarse = FastTokenizer.fit(chunks, FastConfig(scale=2.0, vocab_size=512))
    fine = FastTokenizer.fit(chunks, FastConfig(scale=25.0, vocab_size=512))

    def evaluate(tok):
        toks = tok.encode_batch(chunks[:48])
        err = reconstruction_error(chunks[:48], tok.decode_batch(toks))
        return np.mean([len(t) for t in toks]), err.normalized_rmse

    coarse_tokens, coarse_err = evaluate(coarse)
    fine_tokens, fine_err = evaluate(fine)
    assert fine_tokens > coarse_tokens
    assert fine_err < coarse_err


def test_compression_improves_with_control_frequency():
    """The paper's central claim, at the smallest scale that shows it."""

    def tokens_at(hz):
        c = synthetic_chunks(
            n_chunks=400, horizon=HORIZON, action_dim=ACTION_DIM,
            control_hz=hz, bandwidth_hz=2.0, seed=7,
        )
        tok = FastTokenizer.fit(c, FastConfig(scale=10.0, vocab_size=512))
        return np.mean([len(t) for t in tok.encode_batch(c[:48])])

    assert tokens_at(100.0) < tokens_at(10.0)


def test_step_signals_compress_worse_than_smooth_ones():
    """Gripper-like step edges are the DCT's weak case; that should be visible."""
    steps = gripper_like_chunks(n_chunks=400, horizon=HORIZON, seed=7)
    smooth = synthetic_chunks(
        n_chunks=400, horizon=HORIZON, action_dim=1, control_hz=50.0, seed=7
    )

    def tokens(data):
        tok = FastTokenizer.fit(data, FastConfig(scale=10.0, vocab_size=512))
        return np.mean([len(t) for t in tok.encode_batch(data[:48])])

    assert tokens(steps) > tokens(smooth)


def test_save_load_preserves_encoding(tokenizer, chunks, tmp_path):
    tokenizer.save(tmp_path / "tok")
    loaded = FastTokenizer.load(tmp_path / "tok")
    assert loaded.encode(chunks[0]) == tokenizer.encode(chunks[0])
    assert np.allclose(
        loaded.decode(loaded.encode(chunks[0])), tokenizer.decode(tokenizer.encode(chunks[0]))
    )


def test_decode_survives_a_malformed_token_sequence(tokenizer):
    """Generated sequences can be garbage; decoding must not raise."""
    out = tokenizer.decode([0, 5, 999_999, 3])
    assert out.shape == (HORIZON, ACTION_DIM)
    assert np.isfinite(out).all()


def test_vocab_must_exceed_alphabet(chunks):
    with pytest.raises(ValueError, match="must exceed"):
        FastTokenizer.fit(chunks, FastConfig(scale=200.0, vocab_size=256))


def test_naive_binning_token_count_is_fixed(chunks):
    nb = NaiveBinningTokenizer.fit(chunks, n_bins=256)
    assert len(nb.encode(chunks[0])) == HORIZON * ACTION_DIM
    assert len(nb.encode(chunks[1])) == HORIZON * ACTION_DIM
