import numpy as np
import pytest

from rccn_evolution import (
    EvolutionConfig,
    RCCNParameters,
    create_connection_matrix,
    gen_topology,
    run_evolution,
    simulate_lag_times,
)


def test_simulate_lag_times_shape_and_bounds():
    rng = np.random.default_rng(1)
    topology = gen_topology(40, L_max=12, rng=rng)
    J = create_connection_matrix(topology, rng=rng)
    params = RCCNParameters(T_w=10, equilibration_time=20, relaxation_time=30)

    lags = simulate_lag_times(J, topology, 6, params=params, rng=rng)

    assert lags.shape == (6,)
    assert np.all(lags >= 0)
    assert np.all(lags <= params.relaxation_time)


def test_run_evolution_smoke():
    rng = np.random.default_rng(2)
    topology = gen_topology(50, L_max=12, rng=rng)
    J = create_connection_matrix(topology, rng=rng)
    params = RCCNParameters(T_w=10, equilibration_time=20, relaxation_time=30)
    config = EvolutionConfig(
        bottleneck_size=12,
        carrying_capacity=40,
        mutation_rate=0.05,
        n_cycles=2,
        antibiotic_duration=None,
    )

    result = run_evolution(J, topology, config, params, rng=rng)

    assert len(result.population) == config.bottleneck_size
    assert len(result.history) == config.n_cycles
    assert result.dominant_genotype in result.J_dict


def test_torch_backend_smoke():
    pytest.importorskip("torch")

    rng = np.random.default_rng(3)
    topology = gen_topology(30, L_max=10, rng=rng)
    J = create_connection_matrix(topology, rng=rng)
    params = RCCNParameters(
        T_w=8,
        equilibration_time=15,
        relaxation_time=20,
        backend="torch",
        device="cpu",
    )

    lags = simulate_lag_times(J, topology, 4, params=params, rng=rng)

    assert lags.shape == (4,)
    assert np.all(lags >= 0)
