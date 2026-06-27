from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .rccn import RCCNParameters, j_sigma, simulate_lag_times_batched


@dataclass(frozen=True)
class EvolutionConfig:
    """Parameters for serial-dilution RCCN evolution."""

    bottleneck_size: int = 2000
    carrying_capacity: int = 50000
    growth_rate: float = 0.2
    mutation_rate: float = 5 / (50000 - 2000)
    n_cycles: int = 100
    antibiotic_duration: int | None = None
    antibiotic_survival_floor: float = 0.05
    max_growth_time: int | None = None


@dataclass
class EvolutionResult:
    """Result returned by :func:`run_evolution`."""

    J_dict: dict[int, np.ndarray]
    population: np.ndarray
    history: list[dict]

    @property
    def dominant_genotype(self) -> int:
        return Counter(self.population.tolist()).most_common(1)[0][0]

    @property
    def dominant_J(self) -> np.ndarray:
        return self.J_dict[self.dominant_genotype]


class RowColumnMutator:
    """RCCN mutation: resample one row or column and rescale the matrix."""

    def __init__(self, topology: np.ndarray, gamma: float = 1.5):
        self.topology = np.asarray(topology)
        self.sigma = j_sigma(len(self.topology), gamma=gamma)

    def __call__(self, J: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mutant = np.array(J, copy=True)
        loop_i = rng.integers(len(self.topology))
        new_connections = rng.normal(0.0, self.sigma, size=len(self.topology))
        if rng.random() < 0.5:
            mutant[loop_i, :] = new_connections
        else:
            mutant[:, loop_i] = new_connections

        mutant = mutant - mutant.mean()
        std = mutant.std()
        if std > 0:
            mutant = mutant / std * self.sigma
        np.fill_diagonal(mutant, 0.0)
        return mutant.astype(np.float32)


def _lag_hist(lag_times: np.ndarray, max_t: int) -> np.ndarray:
    alive = lag_times[lag_times >= 0]
    alive = np.minimum(alive, max_t)
    return np.bincount(alive.astype(np.int32), minlength=max_t + 1).astype(np.float32)


def _estimate_growth_time(
    genotypes: np.ndarray,
    lag_times: np.ndarray,
    genotype_ids: list[int],
    growth_rate: float,
    carrying_capacity: int,
) -> int:
    max_t = int(max(0, lag_times.max(initial=0)))
    f = np.stack([_lag_hist(lag_times[genotypes == g], max_t) for g in genotype_ids])
    N = np.zeros(len(genotype_ids), dtype=np.float64)
    N_tot = 0.0
    t = 0
    while N_tot < carrying_capacity:
        recovered = f[:, t] if t < f.shape[1] else 0.0
        N = N * np.exp(growth_rate * (1 - N_tot / carrying_capacity)) + recovered
        N_tot = float(np.ceil(N.sum()))
        t += 1
    return max(1, t)


def logistic_growth(
    genotypes: np.ndarray,
    lag_times: np.ndarray,
    growth_rate: float,
    mutation_rate: float,
    carrying_capacity: int,
    next_genotype_id: int,
    rng: np.random.Generator,
    max_time: int | None = None,
) -> tuple[dict[int, float], dict[int, list[int]], int]:
    """Grow lagged cells to carrying capacity and generate mutant genotypes."""

    alive = lag_times >= 0
    genotypes = genotypes[alive]
    lag_times = lag_times[alive].copy()
    if len(genotypes) == 0:
        raise ValueError("all cells were killed before growth")

    positive = lag_times > 0
    if np.any(positive):
        lag_times[positive] -= lag_times[positive].min()

    genotype_ids = sorted(np.unique(genotypes).tolist())
    T = _estimate_growth_time(
        genotypes, lag_times, genotype_ids, growth_rate, carrying_capacity
    )
    if max_time is not None:
        T = min(T, max_time)

    N = {g: np.zeros(T, dtype=np.float64) for g in genotype_ids}
    f = {
        g: _lag_hist(lag_times[genotypes == g], T)
        for g in genotype_ids
    }
    for g in genotype_ids:
        N[g][0] = f[g][0]

    parent_mutant_dict: dict[int, list[int]] = {}

    for t in range(T - 1):
        N_tot = min(sum(N[g][t] for g in list(N)), carrying_capacity)
        for g in list(N):
            before = N[g][t]
            N[g][t + 1] = before * np.exp(growth_rate * (1 - N_tot / carrying_capacity))
            if t + 1 < len(f[g]):
                N[g][t + 1] += f[g][t + 1]

            divisions = max(0, int(round(N[g][t + 1] - before)))
            n_mutations = rng.binomial(divisions, mutation_rate) if divisions else 0
            if not n_mutations:
                continue

            parent_mutant_dict.setdefault(g, [])
            share = (N[g][t + 1] - before) / (n_mutations + 1)
            for _ in range(n_mutations):
                child = next_genotype_id
                next_genotype_id += 1
                parent_mutant_dict[g].append(child)
                N[child] = np.zeros(T, dtype=np.float64)
                f[child] = np.zeros(T, dtype=np.float64)
                N[g][t + 1] -= share
                N[child][t + 1] += share

    final_counts = {g: float(N[g][-1]) for g in N}
    return final_counts, parent_mutant_dict, next_genotype_id


def _sample_population(
    final_counts: dict[int, float],
    bottleneck_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    genotypes = np.array(list(final_counts), dtype=np.int32)
    weights = np.array([final_counts[g] for g in genotypes], dtype=np.float64)
    weights = weights / weights.sum()
    return rng.choice(genotypes, size=bottleneck_size, replace=True, p=weights)


def run_evolution(
    initial_J: np.ndarray,
    topology: np.ndarray,
    config: EvolutionConfig,
    rccn_params: RCCNParameters,
    rng: np.random.Generator | None = None,
    mutator: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> EvolutionResult:
    """Run serial-dilution evolution starting from a single RCCN genotype."""

    rng = np.random.default_rng() if rng is None else rng
    mutator = RowColumnMutator(topology) if mutator is None else mutator
    J_dict: dict[int, np.ndarray] = {0: np.asarray(initial_J, dtype=np.float32)}
    population = np.zeros(config.bottleneck_size, dtype=np.int32)
    next_genotype_id = 1
    history: list[dict] = []

    for cycle in range(1, config.n_cycles + 1):
        Js = np.stack([J_dict[int(g)] for g in population], axis=0)
        lag_times = simulate_lag_times_batched(
            Js,
            topology,
            params=rccn_params,
            rng=rng,
        )

        if config.antibiotic_duration is not None:
            lag_times = lag_times.copy()
            killed = lag_times < config.antibiotic_duration
            if np.all(killed):
                n_survivors = max(1, int(round(config.antibiotic_survival_floor * len(lag_times))))
                survivor_idx = np.argsort(lag_times)[-n_survivors:]
                killed[survivor_idx] = False
            lag_times[killed] = -1

        final_counts, parent_mutants, next_genotype_id = logistic_growth(
            population,
            lag_times,
            growth_rate=config.growth_rate,
            mutation_rate=config.mutation_rate,
            carrying_capacity=config.carrying_capacity,
            next_genotype_id=next_genotype_id,
            rng=rng,
            max_time=config.max_growth_time,
        )

        for parent, children in parent_mutants.items():
            for child in children:
                J_dict[child] = mutator(J_dict[parent], rng)

        population = _sample_population(final_counts, config.bottleneck_size, rng)
        counts = Counter(population.tolist())
        history.append(
            {
                "cycle": cycle,
                "n_genotypes": len(counts),
                "dominant_genotype": counts.most_common(1)[0][0],
                "dominant_fraction": counts.most_common(1)[0][1] / len(population),
                "mean_lag_before_killing": float(np.mean(lag_times[lag_times >= 0]))
                if np.any(lag_times >= 0)
                else np.nan,
                "n_new_mutants": sum(len(v) for v in parent_mutants.values()),
            }
        )
        if progress_callback is not None:
            progress_callback(history[-1])

    return EvolutionResult(J_dict=J_dict, population=population, history=history)
