"""Sources of action chunks.

Two paths:

- `synthetic_chunks` -- band-limited trajectories with an explicit control
  frequency. Being able to dial control_hz while holding the chunk length fixed
  is what makes the FAST paper's central claim measurable: at higher control
  rates a fixed-length chunk covers less physical motion, so it is smoother, so
  frequency-space tokenization wins by more.
- `load_lerobot_chunks` -- real action chunks from any LeRobot dataset on the
  Hugging Face Hub. Requires the `data` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_HORIZON = 50
DEFAULT_ACTION_DIM = 14  # bimanual ALOHA: 2 x (6 joints + gripper)


def synthetic_chunks(
    n_chunks: int = 2000,
    horizon: int = DEFAULT_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    control_hz: float = 50.0,
    bandwidth_hz: float = 2.0,
    n_components: int = 4,
    noise_std: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    """Generate band-limited pseudo-robot action chunks.

    Each dimension is a sum of `n_components` sinusoids with frequencies drawn
    uniformly below `bandwidth_hz`, sampled at `control_hz` for `horizon` steps.
    Real manipulator trajectories are similarly band-limited -- actuators cannot
    move faster than their bandwidth -- which is the structural fact FAST
    exploits.

    Returns:
        (n_chunks, horizon, action_dim) float64.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(horizon, dtype=np.float64) / control_hz  # (T,)

    freqs = rng.uniform(0.05, bandwidth_hz, size=(n_chunks, action_dim, n_components))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_chunks, action_dim, n_components))
    amps = rng.uniform(0.2, 1.0, size=(n_chunks, action_dim, n_components))
    amps /= amps.sum(axis=-1, keepdims=True)

    # (N, D, C, 1) * (T,) -> broadcast to (N, D, C, T), then sum over components.
    phase = 2.0 * np.pi * freqs[..., None] * t + phases[..., None]
    signal = (amps[..., None] * np.sin(phase)).sum(axis=2)  # (N, D, T)

    chunks = np.transpose(signal, (0, 2, 1))  # (N, T, D)
    if noise_std > 0:
        chunks = chunks + rng.normal(0.0, noise_std, size=chunks.shape)

    # Give dimensions different physical scales, as real robots have. This is
    # what makes per-dimension quantile normalization actually necessary.
    dim_scale = rng.uniform(0.1, 3.0, size=action_dim)
    return chunks * dim_scale


def gripper_like_chunks(
    n_chunks: int = 2000,
    horizon: int = DEFAULT_HORIZON,
    control_hz: float = 50.0,
    seed: int = 0,
) -> np.ndarray:
    """Single-dimension near-binary open/close signals.

    Grippers are the pathological case for DCT: a step edge has energy at every
    frequency, so it does not compress and it rings under quantization. Worth
    keeping visible rather than only ever testing on smooth sinusoids.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_chunks, horizon, 1))
    for i in range(n_chunks):
        n_switches = rng.integers(0, 3)
        edges = np.sort(rng.integers(0, horizon, size=n_switches))
        state = float(rng.integers(0, 2))
        prev = 0
        for e in edges:
            out[i, prev:e, 0] = state
            state = 1.0 - state
            prev = e
        out[i, prev:, 0] = state
    return out


def chunk_trajectory(trajectory: np.ndarray, horizon: int, stride: int = 1) -> np.ndarray:
    """Slice a long (L, D) trajectory into overlapping (T, D) chunks."""
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (L, D) trajectory, got {arr.shape}")
    length = arr.shape[0]
    if length < horizon:
        raise ValueError(f"trajectory of length {length} shorter than horizon {horizon}")
    starts = range(0, length - horizon + 1, stride)
    return np.stack([arr[s : s + horizon] for s in starts])


def load_lerobot_chunks(
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human",
    horizon: int = DEFAULT_HORIZON,
    max_episodes: int = 50,
    stride: int = 25,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Load real action chunks from a LeRobot dataset on the Hub.

    Reads only the parquet action columns -- no video is downloaded, so this
    stays small enough for a laptop. Requires: uv sync --extra data
    """
    try:
        import pandas as pd
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "load_lerobot_chunks needs the 'data' extra: uv sync --extra data"
        ) from exc

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/**/*.parquet", "meta/**"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    files = sorted(Path(local).glob("data/**/*.parquet"))[:max_episodes]
    if not files:
        raise FileNotFoundError(f"no parquet action files found in {repo_id}")

    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=["action"])
        traj = np.stack(df["action"].to_numpy())
        if traj.shape[0] >= horizon:
            chunks.append(chunk_trajectory(traj, horizon, stride))
    if not chunks:
        raise ValueError(f"no episode in {repo_id} was at least {horizon} steps long")
    return np.concatenate(chunks, axis=0)
