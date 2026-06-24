from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np

from rccn_evolution import RCCNParameters, create_connection_matrix, simulate_lag_times


def paper_like_topology(n_spins: int = 16_000, n_loops: int = 345) -> np.ndarray:
    """Make a deterministic topology with paper-like size and loop count."""

    if n_loops <= 0 or n_spins < n_loops:
        raise ValueError("n_spins must be at least n_loops")
    base = n_spins // n_loops
    rem = n_spins % n_loops
    topology = np.full(n_loops, base, dtype=np.int32)
    topology[:rem] += 1
    return topology


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
                "torch_cuda_device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        )
    except Exception as exc:
        info.update({"torch_error": f"{type(exc).__name__}: {exc}"})
    return info


def run_probe(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    topology = paper_like_topology(args.n_spins, args.n_loops)
    J = create_connection_matrix(topology, rng=rng)
    params = RCCNParameters(
        T_w=args.T_w,
        equilibration_time=args.equilibration_time,
        relaxation_time=args.relaxation_time,
        backend=args.backend,
        device=args.device,
    )
    timings = []
    lag_means = []
    lag_p90s = []

    for _ in range(args.repeats):
        t0 = time.perf_counter()
        lags = simulate_lag_times(
            J,
            topology,
            args.n_realizations,
            params=params,
            rng=rng,
        )
        timings.append(time.perf_counter() - t0)
        lag_means.append(float(lags.mean()))
        lag_p90s.append(float(np.percentile(lags, 90)))

    median_seconds = statistics.median(timings)
    paper_realizations = args.paper_realizations
    paper_cycles = args.paper_cycles
    seconds_per_cycle = median_seconds * paper_realizations / args.n_realizations
    total_seconds = seconds_per_cycle * paper_cycles
    return {
        "backend": args.backend,
        "device": args.device,
        "n_spins": args.n_spins,
        "n_loops": args.n_loops,
        "n_realizations": args.n_realizations,
        "paper_realizations": paper_realizations,
        "paper_cycles": paper_cycles,
        "steps": args.equilibration_time + args.T_w + args.relaxation_time,
        "repeats": args.repeats,
        "timings_seconds": [round(t, 4) for t in timings],
        "median_seconds": round(median_seconds, 4),
        "seconds_per_cycle_estimate": round(seconds_per_cycle, 2),
        "hours_for_100_cycles_estimate": round(total_seconds / 3600, 3),
        "mean_lag": round(float(statistics.mean(lag_means)), 3),
        "p90_lag": round(float(statistics.mean(lag_p90s)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "torch"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-spins", type=int, default=16_000)
    parser.add_argument("--n-loops", type=int, default=345)
    parser.add_argument("--T-w", type=int, default=200)
    parser.add_argument("--equilibration-time", type=int, default=4000)
    parser.add_argument("--relaxation-time", type=int, default=4000)
    parser.add_argument("--paper-realizations", type=int, default=2000)
    parser.add_argument("--paper-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = [
        {"kind": "environment", **env_info()},
        {"kind": "probe", **run_probe(args)},
    ]
    text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
