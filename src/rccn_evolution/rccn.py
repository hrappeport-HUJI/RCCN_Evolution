from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import warnings

import numpy as np


@dataclass(frozen=True)
class RCCNParameters:
    """Parameters controlling one RCCN lag-time assay."""

    T_w: int = 200
    equilibration_time: int = 4000
    relaxation_time: int = 4000
    spins_p: float = 0.5
    h: float | None = None
    backend: str = "auto"
    device: str | None = None


def gen_topology(
    n_spins: int,
    alpha: float = 1.5,
    L_min: int = 1,
    L_max: int = 2500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate power-law-distributed RCCN loop lengths summing to ``n_spins``."""

    if n_spins <= 0:
        raise ValueError("n_spins must be positive")

    rng = np.random.default_rng() if rng is None else rng
    topology: list[int] = []
    cur_n_spins = 0

    while cur_n_spins < n_spins:
        u = rng.uniform()
        loop_size = int(
            (
                (L_max ** (1 - alpha) - L_min ** (1 - alpha)) * u
                + L_min ** (1 - alpha)
            )
            ** (1 / (1 - alpha))
        )
        loop_size = max(L_min, loop_size)
        remaining = n_spins - cur_n_spins
        topology.append(min(loop_size, remaining))
        cur_n_spins += topology[-1]

    return np.asarray(topology, dtype=np.int32)


def j_sigma(n_loops: int, gamma: float = 1.5) -> float:
    """Standard deviation used for RCCN inter-loop couplings."""

    if n_loops <= 0:
        raise ValueError("n_loops must be positive")
    return gamma / np.sqrt(n_loops)


def create_connection_matrix(
    topology: np.ndarray,
    rng: np.random.Generator | None = None,
    gamma: float = 1.5,
) -> np.ndarray:
    """Create one RCCN connection matrix with zero diagonal."""

    rng = np.random.default_rng() if rng is None else rng
    n_loops = len(topology)
    J = rng.normal(0.0, j_sigma(n_loops, gamma=gamma), size=(n_loops, n_loops))
    np.fill_diagonal(J, 0.0)
    return J.astype(np.float32)


def create_spins(
    n_realizations: int,
    n_spins: int,
    rng: np.random.Generator | None = None,
    p: float = 0.5,
) -> np.ndarray:
    """Sample binary RCCN spin configurations in {-1, 1}."""

    rng = np.random.default_rng() if rng is None else rng
    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=(n_realizations, n_spins),
        p=[1 - p, p],
    ).astype(np.float32)


def field_strength(topology: np.ndarray) -> float:
    """Stress-field magnitude used in the RCCN work, scaled by network size."""

    return 0.8 / np.sqrt(int(np.sum(topology)) / 2**14)


def _feeding_indices(topology: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return connector and feeder indices for non-contiguous RCCN loop layout."""

    c_idx = np.zeros(len(topology), dtype=np.int32)
    c_idx[1:] = np.cumsum(topology[:-1])
    fc_idx = np.cumsum(topology).astype(np.int32) - 1
    return c_idx, fc_idx


def _rewind_feeders(fc_idx: np.ndarray, topology: np.ndarray) -> None:
    starts = np.zeros(len(topology), dtype=np.int32)
    starts[1:] = np.cumsum(topology[:-1])
    fc_idx -= starts
    fc_idx -= 1
    fc_idx %= topology
    fc_idx += starts


def _loop_magnetization(spins: np.ndarray, topology: np.ndarray) -> np.ndarray:
    out = np.empty((len(spins), len(topology)), dtype=np.float32)
    start = 0
    for loop_i, loop_len in enumerate(topology):
        end = start + int(loop_len)
        out[:, loop_i] = spins[:, start:end].mean(axis=1)
        start = end
    return out


def _torch_device(device: str | None):
    import torch

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _has_torch_accelerator() -> bool:
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            import torch
    except Exception:
        return False
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _loop_magnetization_torch(spins, loop_slices):
    import torch

    return torch.stack(
        [spins[:, start:end].mean(dim=1) for start, end in loop_slices],
        dim=1,
    )


def _torch_tensor(array, *, dtype, device):
    import torch

    try:
        return torch.as_tensor(array, dtype=dtype, device=device)
    except Exception:
        # Older Torch wheels can fail against newer NumPy arrays. Keep a slower
        # list fallback for portability, but prefer zero-copy/as_tensor above.
        return torch.tensor(np.asarray(array).tolist(), dtype=dtype, device=device)


def _run_lag_times_batched_torch(
    Js: np.ndarray,
    spins: np.ndarray,
    topology: np.ndarray,
    H: np.ndarray,
    c_idx: np.ndarray,
    fc_idx: np.ndarray,
    lag_start_time: int,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Run the original batched-J RCCN lag kernel with Torch tensors.

    This mirrors the old RCCN implementation's low-level signature, including
    accepting one J matrix per realization.
    """

    import torch

    Js_np = np.asarray(Js, dtype=np.float32)
    spins_np = np.asarray(spins, dtype=np.float32)
    topology_np = np.asarray(topology, dtype=np.int32)
    H_np = np.asarray(H, dtype=np.float32)
    c_idx_np = np.asarray(c_idx, dtype=np.int64)
    fc_idx_np = np.asarray(fc_idx, dtype=np.int64)

    if Js_np.ndim not in {2, 3}:
        raise ValueError("Js must have shape (n_loops, n_loops) or (n_realizations, n_loops, n_loops)")
    if spins_np.ndim != 2:
        raise ValueError("spins must have shape (n_realizations, n_spins)")
    if Js_np.ndim == 3 and Js_np.shape[0] != spins_np.shape[0]:
        raise ValueError("batched Js and spins must have the same number of realizations")
    if Js_np.shape[-2:] != (len(topology_np), len(topology_np)):
        raise ValueError("Js must have trailing shape (n_loops, n_loops)")
    if spins_np.shape[1] != int(np.sum(topology_np)):
        raise ValueError("spins width must match sum(topology)")

    torch_device = _torch_device(device)
    Js_t = _torch_tensor(Js_np, dtype=torch.float32, device=torch_device)
    spins_t = _torch_tensor(spins_np, dtype=torch.float32, device=torch_device)
    topology_t = _torch_tensor(topology_np, dtype=torch.long, device=torch_device)
    H_t = _torch_tensor(H_np, dtype=torch.float32, device=torch_device)
    c_idx_t = _torch_tensor(c_idx_np, dtype=torch.long, device=torch_device)
    fc_idx_t = _torch_tensor(fc_idx_np, dtype=torch.long, device=torch_device)

    starts_np = np.zeros(len(topology_np), dtype=np.int64)
    starts_np[1:] = np.cumsum(topology_np[:-1])
    starts = _torch_tensor(starts_np, dtype=torch.long, device=torch_device)

    n_realizations = spins_t.shape[0]
    lag_times = torch.full(
        (n_realizations,),
        int(len(H_np) - lag_start_time),
        dtype=torch.long,
        device=torch_device,
    )
    active_original_idx = torch.arange(n_realizations, dtype=torch.long, device=torch_device)

    loop_slices = []
    start = 0
    for loop_len in topology_np:
        end = start + int(loop_len)
        loop_slices.append((start, end))
        start = end

    for t in range(len(H_np)):
        if t > lag_start_time and len(spins_t):
            mean_mag = _loop_magnetization_torch(spins_t, loop_slices).mean(dim=1)
            recovered = mean_mag <= 0
            n_recovered = torch.count_nonzero(recovered)
            if n_recovered:
                lag_times[active_original_idx[recovered]] = t - lag_start_time
                if int(n_recovered.item()) == len(spins_t):
                    return np.asarray(lag_times.cpu().tolist(), dtype=np.int32)

                keep = ~recovered
                spins_t = spins_t[keep]
                active_original_idx = active_original_idx[keep]
                if Js_t.ndim == 3:
                    Js_t = Js_t[keep]

        connector = spins_t[:, c_idx_t]
        if Js_t.ndim == 3:
            coupling = torch.bmm(Js_t, connector[:, :, None]).squeeze(-1)
        else:
            coupling = connector @ Js_t.T
        spins_t[:, fc_idx_t] += coupling + H_t[t]
        spins_t[:, fc_idx_t] = torch.sign(spins_t[:, fc_idx_t])
        c_idx_t = fc_idx_t.clone()
        fc_idx_t = (fc_idx_t - starts - 1) % topology_t + starts

    return np.asarray(lag_times.cpu().tolist(), dtype=np.int32)


def simulate_lag_times_batched_torch(
    Js: np.ndarray,
    topology: np.ndarray,
    params: RCCNParameters | None = None,
    rng: np.random.Generator | None = None,
    initial_spins: np.ndarray | None = None,
) -> np.ndarray:
    """Torch/GPU lag simulation for one J matrix per realization."""

    params = RCCNParameters() if params is None else params
    rng = np.random.default_rng() if rng is None else rng
    topology_np = np.asarray(topology, dtype=np.int32)
    Js_np = np.asarray(Js, dtype=np.float32)

    if Js_np.ndim != 3:
        raise ValueError("Js must have shape (n_realizations, n_loops, n_loops)")
    if Js_np.shape[1:] != (len(topology_np), len(topology_np)):
        raise ValueError("Js must have shape (n_realizations, n_loops, n_loops)")

    n_realizations = int(Js_np.shape[0])
    if n_realizations <= 0:
        return np.empty(0, dtype=np.int32)

    n_spins = int(np.sum(topology_np))
    spins = (
        np.asarray(initial_spins, dtype=np.float32)
        if initial_spins is not None
        else create_spins(n_realizations, n_spins, rng=rng, p=params.spins_p)
    )
    h = field_strength(topology_np) if params.h is None else params.h
    lag_start_time = params.equilibration_time + params.T_w
    sim_len = lag_start_time + params.relaxation_time
    H = np.zeros((sim_len, len(topology_np)), dtype=np.float32)
    H[params.equilibration_time:lag_start_time, :] = h
    c_idx, fc_idx = _feeding_indices(topology_np)
    return _run_lag_times_batched_torch(
        Js_np,
        spins,
        topology_np,
        H,
        c_idx,
        fc_idx,
        lag_start_time,
        device=params.device,
    )


def _run_lag_times_batched_numpy(
    Js: np.ndarray,
    spins: np.ndarray,
    topology: np.ndarray,
    H: np.ndarray,
    c_idx: np.ndarray,
    fc_idx: np.ndarray,
    lag_start_time: int,
) -> np.ndarray:
    Js_active = np.asarray(Js, dtype=np.float32)
    spins_active = np.asarray(spins, dtype=np.float32)
    topology_np = np.asarray(topology, dtype=np.int32)
    lag_times = np.full(len(spins_active), len(H) - lag_start_time, dtype=np.int32)
    active_original_idx = np.arange(len(spins_active))

    for t in range(len(H)):
        if t > lag_start_time and len(spins_active):
            mean_mag = _loop_magnetization(spins_active, topology_np).mean(axis=1)
            recovered = mean_mag <= 0
            if np.any(recovered):
                lag_times[active_original_idx[recovered]] = t - lag_start_time
                keep = ~recovered
                if not np.any(keep):
                    break
                spins_active = spins_active[keep]
                Js_active = Js_active[keep]
                active_original_idx = active_original_idx[keep]

        connector = spins_active[:, c_idx]
        coupling = np.einsum("nij,nj->ni", Js_active, connector, optimize=True)
        spins_active[:, fc_idx] += coupling + H[t]
        spins_active[:, fc_idx] = np.sign(spins_active[:, fc_idx])
        c_idx = fc_idx.copy()
        _rewind_feeders(fc_idx, topology_np)

    return lag_times


def simulate_lag_times_batched(
    Js: np.ndarray,
    topology: np.ndarray,
    params: RCCNParameters | None = None,
    rng: np.random.Generator | None = None,
    initial_spins: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate lag times for a population with one J matrix per realization."""

    params = RCCNParameters() if params is None else params
    if params.backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be one of: 'auto', 'numpy', 'torch'")
    if params.backend == "torch" or (
        params.backend == "auto" and (params.device is not None or _has_torch_accelerator())
    ):
        return simulate_lag_times_batched_torch(
            Js,
            topology,
            params=params,
            rng=rng,
            initial_spins=initial_spins,
        )

    rng = np.random.default_rng() if rng is None else rng
    topology_np = np.asarray(topology, dtype=np.int32)
    Js_np = np.asarray(Js, dtype=np.float32)
    if Js_np.ndim != 3:
        raise ValueError("Js must have shape (n_realizations, n_loops, n_loops)")
    if Js_np.shape[1:] != (len(topology_np), len(topology_np)):
        raise ValueError("Js must have shape (n_realizations, n_loops, n_loops)")
    if Js_np.shape[0] <= 0:
        return np.empty(0, dtype=np.int32)

    n_spins = int(np.sum(topology_np))
    spins = (
        np.asarray(initial_spins, dtype=np.float32)
        if initial_spins is not None
        else create_spins(Js_np.shape[0], n_spins, rng=rng, p=params.spins_p)
    )
    h = field_strength(topology_np) if params.h is None else params.h
    lag_start_time = params.equilibration_time + params.T_w
    sim_len = lag_start_time + params.relaxation_time
    H = np.zeros((sim_len, len(topology_np)), dtype=np.float32)
    H[params.equilibration_time:lag_start_time, :] = h
    c_idx, fc_idx = _feeding_indices(topology_np)
    return _run_lag_times_batched_numpy(
        Js_np,
        spins,
        topology_np,
        H,
        c_idx,
        fc_idx,
        lag_start_time,
    )


def simulate_lag_times_torch(
    J: np.ndarray,
    topology: np.ndarray,
    n_realizations: int,
    params: RCCNParameters | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Torch/GPU implementation of :func:`simulate_lag_times`.

    The function uses CUDA or MPS when available, and otherwise runs on Torch CPU.
    Random initial spins are generated by NumPy so callers can use the same RNG
    interface as the NumPy backend.
    """

    import torch

    if n_realizations <= 0:
        return np.empty(0, dtype=np.int32)

    params = RCCNParameters() if params is None else params
    rng = np.random.default_rng() if rng is None else rng
    topology_np = np.asarray(topology, dtype=np.int32)
    J_np = np.asarray(J, dtype=np.float32)

    if J_np.shape != (len(topology_np), len(topology_np)):
        raise ValueError("J must have shape (n_loops, n_loops)")

    device = _torch_device(params.device)
    n_spins = int(np.sum(topology_np))
    h = field_strength(topology_np) if params.h is None else params.h
    lag_start_time = params.equilibration_time + params.T_w
    sim_len = lag_start_time + params.relaxation_time

    # Use Python-list conversion to avoid relying on torch's NumPy bridge. Some
    # older Torch wheels are compiled against NumPy 1.x and fail against NumPy 2.
    J_t = torch.tensor(J_np.tolist(), dtype=torch.float32, device=device)
    topology_t = torch.tensor(topology_np.tolist(), dtype=torch.long, device=device)
    c_idx_np, fc_idx_np = _feeding_indices(topology_np)
    c_idx = torch.tensor(c_idx_np.tolist(), dtype=torch.long, device=device)
    fc_idx = torch.tensor(fc_idx_np.tolist(), dtype=torch.long, device=device)
    starts_np = np.zeros(len(topology_np), dtype=np.int32)
    starts_np[1:] = np.cumsum(topology_np[:-1])
    starts = torch.tensor(starts_np.tolist(), dtype=torch.long, device=device)

    spins_np = create_spins(n_realizations, n_spins, rng=rng, p=params.spins_p)
    spins = torch.tensor(spins_np.tolist(), dtype=torch.float32, device=device)
    active_original_idx = torch.arange(n_realizations, dtype=torch.long, device=device)
    lag_times = torch.full(
        (n_realizations,),
        params.relaxation_time,
        dtype=torch.long,
        device=device,
    )
    loop_slices = []
    start = 0
    for loop_len in topology_np:
        end = start + int(loop_len)
        loop_slices.append((start, end))
        start = end

    for t in range(sim_len):
        if t > lag_start_time and len(spins):
            mean_mag = _loop_magnetization_torch(spins, loop_slices).mean(dim=1)
            recovered = mean_mag <= 0
            if torch.any(recovered):
                lag_times[active_original_idx[recovered]] = t - lag_start_time
                keep = ~recovered
                spins = spins[keep]
                active_original_idx = active_original_idx[keep]
                if len(spins) == 0:
                    break

        field = h if params.equilibration_time <= t < lag_start_time else 0.0
        inputs = spins[:, fc_idx] + spins[:, c_idx] @ J_t.T + field
        spins[:, fc_idx] = torch.sign(inputs)
        c_idx = fc_idx.clone()
        fc_idx = (fc_idx - starts - 1) % topology_t + starts

    return np.asarray(lag_times.cpu().tolist(), dtype=np.int32)


def simulate_lag_times(
    J: np.ndarray,
    topology: np.ndarray,
    n_realizations: int,
    params: RCCNParameters | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate RCCN lag times for one genotype.

    Lag is measured after field removal as the first time the mean loop
    magnetization crosses zero, matching the practical criterion used by the
    original RCCN scripts.
    """

    params = RCCNParameters() if params is None else params
    if params.backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be one of: 'auto', 'numpy', 'torch'")
    if params.backend == "torch" or (
        params.backend == "auto" and (params.device is not None or _has_torch_accelerator())
    ):
        return simulate_lag_times_torch(J, topology, n_realizations, params=params, rng=rng)

    if n_realizations <= 0:
        return np.empty(0, dtype=np.int32)

    rng = np.random.default_rng() if rng is None else rng
    topology = np.asarray(topology, dtype=np.int32)
    J = np.asarray(J, dtype=np.float32)

    if J.shape != (len(topology), len(topology)):
        raise ValueError("J must have shape (n_loops, n_loops)")

    n_spins = int(np.sum(topology))
    h = field_strength(topology) if params.h is None else params.h
    lag_start_time = params.equilibration_time + params.T_w
    sim_len = lag_start_time + params.relaxation_time

    spins = create_spins(n_realizations, n_spins, rng=rng, p=params.spins_p)
    active_original_idx = np.arange(n_realizations)
    lag_times = np.full(n_realizations, params.relaxation_time, dtype=np.int32)
    c_idx, fc_idx = _feeding_indices(topology)

    for t in range(sim_len):
        if t > lag_start_time and len(spins):
            mean_mag = _loop_magnetization(spins, topology).mean(axis=1)
            recovered = mean_mag <= 0
            if np.any(recovered):
                lag_times[active_original_idx[recovered]] = t - lag_start_time
                keep = ~recovered
                spins = spins[keep]
                active_original_idx = active_original_idx[keep]
                if len(spins) == 0:
                    break

        field = h if params.equilibration_time <= t < lag_start_time else 0.0
        inputs = spins[:, fc_idx] + spins[:, c_idx] @ J.T + field
        spins[:, fc_idx] = np.sign(inputs)
        c_idx = fc_idx.copy()
        _rewind_feeders(fc_idx, topology)

    return lag_times
