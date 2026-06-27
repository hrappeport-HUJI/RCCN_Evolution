from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from rccn_evolution import (
    EvolutionConfig,
    RCCNParameters,
    create_connection_matrix,
    gen_topology,
    run_evolution,
)
from rccn_evolution.rccn import _run_lag_times_batched_torch


OLD_RCCN_SIM = Path("/Users/hrappeport/PycharmProjects/ScriptHub/ltee/RCCN_sim.py")


def _load_old_rccn_sim():
    if not OLD_RCCN_SIM.exists():
        pytest.skip(f"old RCCN_sim.py not found at {OLD_RCCN_SIM}")
    spec = importlib.util.spec_from_file_location("old_rccn_sim_for_tests", OLD_RCCN_SIM)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not import old RCCN_sim.py from {OLD_RCCN_SIM}")
    module = importlib.util.module_from_spec(spec)
    old_numba = sys.modules.get("numba")
    old_numba_cuda = sys.modules.get("numba.cuda")
    cuda_stub = types.SimpleNamespace(
        jit=lambda f=None, **_: f if f is not None else (lambda fn: fn),
        as_cuda_array=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cuda stub is only for importing old CPU test path")
        ),
    )
    sys.modules["numba"] = types.SimpleNamespace(cuda=cuda_stub)
    sys.modules["numba.cuda"] = cuda_stub
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"old RCCN_sim.py could not be imported: {exc}")
    finally:
        if old_numba is None:
            sys.modules.pop("numba", None)
        else:
            sys.modules["numba"] = old_numba
        if old_numba_cuda is None:
            sys.modules.pop("numba.cuda", None)
        else:
            sys.modules["numba.cuda"] = old_numba_cuda
    module.DEVICE_NAME = "cpu"
    return module


def _fixed_problem(n_realizations: int, n_distinct_j: int):
    old = _load_old_rccn_sim()
    topology = np.asarray([3, 4, 2, 5, 3], dtype=np.int32)
    n_loops = len(topology)
    n_spins = int(np.sum(topology))
    params = RCCNParameters(
        T_w=7,
        equilibration_time=11,
        relaxation_time=18,
        backend="torch",
        device="cpu",
    )
    rng = np.random.default_rng(12345)
    base_js = np.stack(
        [create_connection_matrix(topology, rng=rng) for _ in range(n_distinct_j)]
    )
    js = base_js[np.arange(n_realizations) % n_distinct_j].astype(np.float32)

    np.random.seed(2468)
    spins = old.create_spins(
        spins=None,
        same_init_spins=False,
        spins_p=params.spins_p,
        n_realizations=n_realizations,
        n_spins=n_spins,
    )

    sim_len = params.equilibration_time + params.T_w + params.relaxation_time
    lag_start_time = params.equilibration_time + params.T_w
    h = 0.8 / np.sqrt(n_spins / 2**14)
    H = np.zeros((sim_len, n_loops), dtype=np.float32)
    H[params.equilibration_time:lag_start_time, :] = h
    c_idx, _, fc_idx, _ = old.get_feeding_idx(
        topology, contiguous_C=False, return_only_F=False
    )
    return old, topology, js, spins, H, c_idx, fc_idx, lag_start_time


def _torch_tensor(array, *, dtype, device):
    import torch

    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _old_lag_times(old, topology, js, spins, H, c_idx, fc_idx, lag_start_time, device="cpu"):
    import torch

    torch_device = torch.device(device)
    old.DEVICE_NAME = "cuda:0" if torch_device.type == "cuda" else "cpu"
    return np.asarray(
        old._run_RCCN_lag_times(
            J_C=_torch_tensor(js.copy(), dtype=torch.float32, device=torch_device),
            spins=_torch_tensor(spins.copy(), dtype=torch.float32, device=torch_device),
            topology=_torch_tensor(topology.copy(), dtype=torch.long, device=torch_device),
            H=_torch_tensor(H.copy(), dtype=torch.float32, device=torch_device),
            C_idx=_torch_tensor(c_idx.copy(), dtype=torch.long, device=torch_device),
            FC_idx=_torch_tensor(fc_idx.copy(), dtype=torch.long, device=torch_device),
            lag_start_time=lag_start_time,
        )
        .detach()
        .cpu()
        .tolist(),
        dtype=np.int32,
    )


def _new_lag_times(topology, js, spins, H, c_idx, fc_idx, lag_start_time, device="cpu"):
    return _run_lag_times_batched_torch(
        js.copy(),
        spins.copy(),
        topology.copy(),
        H.copy(),
        c_idx.copy(),
        fc_idx.copy(),
        lag_start_time,
        device=device,
    )


def test_batched_kernel_matches_old_code_for_mixed_genotypes():
    old, topology, js, spins, H, c_idx, fc_idx, lag_start_time = _fixed_problem(
        n_realizations=12,
        n_distinct_j=3,
    )

    old_lags = _old_lag_times(old, topology, js, spins, H, c_idx, fc_idx, lag_start_time)
    new_lags = _new_lag_times(topology, js, spins, H, c_idx, fc_idx, lag_start_time)

    assert np.array_equal(new_lags, old_lags)


def test_batched_kernel_matches_old_code_for_repeated_single_j():
    old, topology, js, spins, H, c_idx, fc_idx, lag_start_time = _fixed_problem(
        n_realizations=10,
        n_distinct_j=1,
    )

    old_lags = _old_lag_times(old, topology, js, spins, H, c_idx, fc_idx, lag_start_time)
    new_lags = _new_lag_times(topology, js, spins, H, c_idx, fc_idx, lag_start_time)

    assert np.array_equal(new_lags, old_lags)


def test_run_evolution_calls_batched_lag_simulation_once_per_cycle(monkeypatch):
    import rccn_evolution.evolution as evolution

    calls = []

    def fake_batched_lags(js, topology, params=None, rng=None):
        calls.append(js.shape)
        return np.zeros(js.shape[0], dtype=np.int32)

    def fail_single_j_lags(*args, **kwargs):
        raise AssertionError("single-genotype lag simulation should not be used")

    monkeypatch.setattr(evolution, "simulate_lag_times_batched", fake_batched_lags)
    monkeypatch.setattr(evolution, "simulate_lag_times", fail_single_j_lags, raising=False)

    rng = np.random.default_rng(9)
    topology = gen_topology(40, L_max=12, rng=rng)
    initial_j = create_connection_matrix(topology, rng=rng)
    config = EvolutionConfig(
        bottleneck_size=8,
        carrying_capacity=20,
        mutation_rate=0.0,
        n_cycles=4,
        antibiotic_duration=None,
    )
    params = RCCNParameters(
        T_w=3,
        equilibration_time=5,
        relaxation_time=6,
        backend="torch",
        device="cpu",
    )

    run_evolution(initial_j, topology, config, params, rng=rng)

    assert calls == [
        (config.bottleneck_size, len(topology), len(topology))
        for _ in range(config.n_cycles)
    ]


@pytest.mark.skipif(os.environ.get("RCCN_BENCH") != "1", reason="set RCCN_BENCH=1 to run")
def test_batched_kernel_runtime_is_at_least_as_fast_as_old_code():
    import torch

    old, topology, js, spins, H, c_idx, fc_idx, lag_start_time = _fixed_problem(
        n_realizations=int(os.environ.get("RCCN_BENCH_N", "128")),
        n_distinct_j=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def old_run():
        return _old_lag_times(old, topology, js, spins, H, c_idx, fc_idx, lag_start_time, device=device)

    def new_run():
        return _new_lag_times(topology, js, spins, H, c_idx, fc_idx, lag_start_time, device=device)

    assert np.array_equal(new_run(), old_run())

    old_run()
    new_run()
    old_times = []
    new_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        old_run()
        old_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        new_run()
        new_times.append(time.perf_counter() - t0)

    old_median = float(np.median(old_times))
    new_median = float(np.median(new_times))
    print(
        {
            "old_median_sec": old_median,
            "new_median_sec": new_median,
            "speedup": old_median / new_median if new_median else float("inf"),
            "bottleneck_size": len(js),
            "n_loops": len(topology),
            "n_spins": int(np.sum(topology)),
            "device": device,
        }
    )
    assert new_median <= old_median * 1.05
