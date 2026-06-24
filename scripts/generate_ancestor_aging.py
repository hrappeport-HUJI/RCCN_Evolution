from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from rccn_evolution import (
    RCCNParameters,
    create_connection_matrix,
    simulate_lag_times,
)


def parse_t_ws(text: str) -> list[int]:
    return [int(float(x)) for x in text.split(",") if x.strip()]


def one_minus_cdf(lag_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(lag_times)
    ys = 1 - np.linspace(0, 1, len(xs), endpoint=False)
    return xs, ys


def plot_one_minus_cdfs(
    lag_times_by_tw: dict[int, np.ndarray],
    ax,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    cmap = colors.LinearSegmentedColormap(
        "rccn_tw",
        {
            "red": ((0.0, 0.22, 0.0), (0.5, 1.0, 1.0), (1.0, 0.89, 1.0)),
            "green": ((0.0, 0.49, 0.0), (0.5, 1.0, 1.0), (1.0, 0.12, 1.0)),
            "blue": ((0.0, 0.72, 0.0), (0.5, 0.0, 0.0), (1.0, 0.11, 1.0)),
        },
    )
    t_ws = sorted(lag_times_by_tw)
    for color_i, t_w in zip(np.linspace(0.25, 1, len(t_ws)), t_ws):
        xs, ys = one_minus_cdf(lag_times_by_tw[t_w])
        ax.plot(xs, ys, color=cmap(color_i), lw=1.8, label=f"T_w={t_w}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Recovery lag")
    ax.set_ylabel("1-CDF")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.2)


def save_plots(
    all_lags: dict[int, dict[int, np.ndarray]],
    out_dir: Path,
) -> None:
    max_lag = max(float(np.max(lags)) for by_tw in all_lags.values() for lags in by_tw.values())
    xlim = (0, max(50, max_lag * 1.05))

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.6), sharex=True, sharey=True)
    for anc_i, ax in zip(sorted(all_lags), axes.flat):
        plot_one_minus_cdfs(all_lags[anc_i], ax, title=f"Ancestor J {anc_i}", xlim=xlim)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Ancestor RCCN aging: lag 1-CDFs", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / "ancestor_aging_1cdf_grid.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "ancestor_aging_1cdf_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    for anc_i in sorted(all_lags):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_one_minus_cdfs(all_lags[anc_i], ax, title=f"Ancestor J {anc_i}", xlim=xlim)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"ancestor_{anc_i:02d}_aging_1cdf.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, default=Path("data/topology_1.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ancestor_aging"))
    parser.add_argument("--n-ancestors", type=int, default=10)
    parser.add_argument("--n-realizations", type=int, default=2000)
    parser.add_argument("--t-ws", default="20,63,200,632,2000")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--equilibration-time", type=int, default=4000)
    parser.add_argument("--relaxation-time", type=int, default=4000)
    args = parser.parse_args()

    out_dir = args.out_dir
    lag_dir = out_dir / "lag_times"
    j_dir = out_dir / "J_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    lag_dir.mkdir(exist_ok=True)
    j_dir.mkdir(exist_ok=True)

    topology = np.load(args.topology)
    t_ws = parse_t_ws(args.t_ws)
    rng = np.random.default_rng(args.seed)

    metadata = {
        "topology_path": str(args.topology),
        "n_spins": int(np.sum(topology)),
        "n_loops": int(len(topology)),
        "n_ancestors": args.n_ancestors,
        "n_realizations": args.n_realizations,
        "t_ws": t_ws,
        "seed": args.seed,
        "backend": args.backend,
        "device": args.device,
        "equilibration_time": args.equilibration_time,
        "relaxation_time": args.relaxation_time,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    rows = []
    all_lags: dict[int, dict[int, np.ndarray]] = {}
    for anc_i in range(args.n_ancestors):
        j_path = j_dir / f"ancestor_J_{anc_i:02d}.npy"
        if j_path.exists():
            J = np.load(j_path)
        else:
            J = create_connection_matrix(topology, rng=rng)
            np.save(j_path, J)

        all_lags[anc_i] = {}
        for t_w in t_ws:
            lag_path = lag_dir / f"ancestor_{anc_i:02d}_T_w_{t_w}.npy"
            if lag_path.exists():
                lags = np.load(lag_path)
            else:
                params = RCCNParameters(
                    T_w=t_w,
                    equilibration_time=args.equilibration_time,
                    relaxation_time=args.relaxation_time,
                    backend=args.backend,
                    device=args.device,
                )
                lags = simulate_lag_times(
                    J,
                    topology,
                    args.n_realizations,
                    params=params,
                    rng=rng,
                )
                np.save(lag_path, lags)
            all_lags[anc_i][t_w] = lags
            rows.append(
                {
                    "ancestor": anc_i,
                    "T_w": t_w,
                    "mean": float(np.mean(lags)),
                    "median": float(np.median(lags)),
                    "p90": float(np.percentile(lags, 90)),
                    "p99": float(np.percentile(lags, 99)),
                    "max": int(np.max(lags)),
                }
            )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_plots(all_lags, out_dir)
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
