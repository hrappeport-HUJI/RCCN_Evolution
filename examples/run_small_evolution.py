from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rccn_evolution import (
    EvolutionConfig,
    RCCNParameters,
    create_connection_matrix,
    gen_topology,
    run_evolution,
    simulate_lag_times,
)


def main() -> None:
    rng = np.random.default_rng(7)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    # Small demo values. Paper-scale runs used 16,000 spins and 100 cycles.
    topology = gen_topology(n_spins=160, L_max=40, rng=rng)
    initial_J = create_connection_matrix(topology, rng=rng)

    rccn_params = RCCNParameters(
        T_w=40,
        equilibration_time=150,
        relaxation_time=250,
    )
    config = EvolutionConfig(
        bottleneck_size=80,
        carrying_capacity=800,
        growth_rate=0.2,
        mutation_rate=0.02,
        n_cycles=8,
        antibiotic_duration=35,
    )

    before = simulate_lag_times(initial_J, topology, 120, params=rccn_params, rng=rng)
    result = run_evolution(initial_J, topology, config, rccn_params, rng=rng)
    after = simulate_lag_times(result.dominant_J, topology, 120, params=rccn_params, rng=rng)

    np.save(out_dir / "evolved_J.npy", result.dominant_J)
    with (out_dir / "final_population.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.history[-1].keys())
        writer.writeheader()
        writer.writerows(result.history)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, rccn_params.relaxation_time + 5, 5)
    ax.hist(before, bins=bins, alpha=0.55, density=True, label="initial")
    ax.hist(after, bins=bins, alpha=0.55, density=True, label="evolved dominant")
    ax.axvline(config.antibiotic_duration, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("RCCN lag time")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "lag_distributions.png", dpi=200)

    print(f"Dominant genotype: {result.dominant_genotype}")
    print(f"Dominant fraction: {result.history[-1]['dominant_fraction']:.3f}")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

