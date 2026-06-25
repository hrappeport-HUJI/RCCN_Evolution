from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rccn_evolution import EvolutionConfig, RCCNParameters, run_evolution


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _log(line: str, log_file) -> None:
    print(line, flush=True)
    print(line, file=log_file, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancestor-j", type=Path, required=True)
    parser.add_argument("--topology", type=Path, default=Path("data/topology_1.npy"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-cycles", type=int, default=200)
    parser.add_argument("--T-a", type=int, default=0)
    parser.add_argument("--T-2", type=int, default=200)
    parser.add_argument("--bottleneck-size", type=int, default=2000)
    parser.add_argument("--carrying-capacity", type=int, default=50000)
    parser.add_argument("--growth-rate", type=float, default=0.2)
    parser.add_argument("--mutation-rate", type=float, default=5 / (50000 - 2000))
    parser.add_argument("--equilibration-time", type=int, default=4000)
    parser.add_argument("--relaxation-time", type=int, default=4000)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"

    topology = np.load(args.topology)
    ancestor_J = np.load(args.ancestor_j)
    rng = np.random.default_rng(args.seed)

    rccn_params = RCCNParameters(
        T_w=args.T_2,
        equilibration_time=args.equilibration_time,
        relaxation_time=args.relaxation_time,
        backend=args.backend,
        device=args.device,
    )
    config = EvolutionConfig(
        bottleneck_size=args.bottleneck_size,
        carrying_capacity=args.carrying_capacity,
        growth_rate=args.growth_rate,
        mutation_rate=args.mutation_rate,
        n_cycles=args.n_cycles,
        antibiotic_duration=args.T_a,
    )

    metadata = {
        "ancestor_j": str(args.ancestor_j),
        "topology": str(args.topology),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "n_cycles": args.n_cycles,
        "T_a": args.T_a,
        "T_2": args.T_2,
        "rccn_params": rccn_params.__dict__,
        "evolution_config": config.__dict__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    started = time.monotonic()
    with progress_path.open("w") as log_file:
        _log(json.dumps({"event": "start", **metadata}, sort_keys=True), log_file)

        def on_progress(row: dict) -> None:
            elapsed = time.monotonic() - started
            cycle = int(row["cycle"])
            seconds_per_cycle = elapsed / cycle
            remaining = max(0, args.n_cycles - cycle) * seconds_per_cycle
            payload = {
                "event": "cycle_complete",
                "cycle": cycle,
                "n_cycles": args.n_cycles,
                "elapsed_sec": round(elapsed, 3),
                "seconds_per_cycle": round(seconds_per_cycle, 3),
                "eta_sec": round(remaining, 3),
                **row,
            }
            _log(json.dumps(payload, sort_keys=True), log_file)

        result = run_evolution(
            ancestor_J,
            topology,
            config,
            rccn_params,
            rng=rng,
            progress_callback=on_progress,
        )

        elapsed = time.monotonic() - started
        final_counts = Counter(result.population.tolist())
        np.save(out_dir / "dominant_J.npy", result.dominant_J)
        np.save(out_dir / "final_population.npy", result.population)
        _write_history(out_dir / "history.csv", result.history)
        with (out_dir / "final_population_counts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["genotype", "count"])
            writer.writeheader()
            for genotype, count in final_counts.most_common():
                writer.writerow({"genotype": genotype, "count": count})

        _log(
            json.dumps(
                {
                    "event": "done",
                    "elapsed_sec": round(elapsed, 3),
                    "dominant_genotype": result.dominant_genotype,
                    "dominant_fraction": result.history[-1]["dominant_fraction"],
                    "n_genotypes_final": len(final_counts),
                },
                sort_keys=True,
            ),
            log_file,
        )


if __name__ == "__main__":
    main()
