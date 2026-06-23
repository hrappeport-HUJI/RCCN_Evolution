# RCCN Evolution

Code for simulating Randomly Connected Cycles Networks (RCCNs) and evolving them under serial dilution with optional antibiotic exposure.

This repository accompanies the RCCN in-silico evolution described in:

> Adaptive Fragility: Evolution of Antibiotic Tolerance by Regulatory Network Disruption

The paper-scale simulations used networks with 345 loops, 16,000 spins, population size 2,000, carrying capacity 50,000, growth rate 0.2, starvation duration `T_w = 200`, and 100 evolution cycles. The example included here uses much smaller defaults so it can run quickly on a laptop.

## Install

```bash
python -m pip install -e ".[dev]"
```

For GPU acceleration, install the optional Torch dependency:

```bash
python -m pip install -e ".[gpu]"
```

## Run A Small Evolution Example

```bash
python examples/run_small_evolution.py
```

The script evolves a small RCCN population for a few serial-dilution cycles and writes:

```text
examples/out/lag_distributions.png
examples/out/final_population.csv
examples/out/evolved_J.npy
```

## Conceptual Model

Each RCCN is a genotype represented by a connection matrix `J`. A genotype induces a lag-time distribution by:

1. Sampling random spin configurations.
2. Equilibrating the RCCN.
3. Applying a homogeneous stress field for duration `T_w`.
4. Removing the field and measuring recovery lag as the first return/crossing of the network magnetization.

Evolution proceeds by repeated cycles:

1. Sample lag times for the current population.
2. Kill cells that recover before the antibiotic exposure ends, if antibiotic is enabled.
3. Grow surviving cells to carrying capacity using logistic growth.
4. Generate mutants during growth.
5. Mutate RCCN connection matrices by resampling one row or column.
6. Randomly bottleneck back to the starting population size.

## Package Layout

```text
src/rccn_evolution/rccn.py       RCCN topology, dynamics, and lag simulation
src/rccn_evolution/evolution.py  Serial-dilution growth and evolution loop
examples/run_small_evolution.py  Minimal runnable example
tests/                          Smoke tests for core behavior
```

## Notes

The package includes both a NumPy reference backend and a Torch backend for GPU acceleration. By default, RCCN lag simulation uses `backend="auto"`: CUDA is used when available, then Apple MPS when available, and otherwise NumPy is used. You can force a backend/device with:

```python
params = RCCNParameters(
    T_w=200,
    backend="torch",
    device="cuda",  # or "mps", "cpu"
)
```

Paper-scale simulations should use the Torch backend.
