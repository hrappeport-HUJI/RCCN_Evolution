from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from rccn_evolution import RCCNParameters, simulate_lag_times


@dataclass(frozen=True)
class Strain:
    name: str
    j_path: Path
    kind: str


def parse_t_ws(text: str) -> list[int]:
    return [int(float(x)) for x in text.split(",") if x.strip()]


def parse_strain(text: str) -> Strain:
    if "=" not in text:
        raise argparse.ArgumentTypeError("strain must be formatted as NAME=PATH")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("strain name cannot be empty")
    return Strain(name=name, j_path=Path(path), kind="evolved")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(p) for p in parts)]).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def one_minus_cdf(lag_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(lag_times)
    ys = 1 - np.linspace(0, 1, len(xs), endpoint=False)
    return xs, ys


def finite_summary(lags: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(lags)),
        "median": float(np.median(lags)),
        "p90": float(np.percentile(lags, 90)),
        "p95": float(np.percentile(lags, 95)),
        "p99": float(np.percentile(lags, 99)),
        "max": int(np.max(lags)),
    }


def load_or_simulate(
    *,
    J: np.ndarray,
    topology: np.ndarray,
    strain: Strain,
    t_w: int,
    n_realizations: int,
    lag_dir: Path,
    seed: int,
    backend: str,
    device: str,
    equilibration_time: int,
    relaxation_time: int,
) -> np.ndarray:
    lag_path = lag_dir / f"{strain.name}_Tw_{t_w}_n_{n_realizations}.npy"
    if lag_path.exists():
        return np.load(lag_path)

    params = RCCNParameters(
        T_w=t_w,
        equilibration_time=equilibration_time,
        relaxation_time=relaxation_time,
        backend=backend,
        device=device,
    )
    rng = np.random.default_rng(stable_seed(seed, strain.name, t_w, n_realizations))
    lags = simulate_lag_times(J, topology, n_realizations, params=params, rng=rng)
    np.save(lag_path, lags)
    return lags


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_lag_vs_tw(
    rows: list[dict],
    strains: list[Strain],
    out_dir: Path,
    train_t_w: int,
) -> None:
    by_name = {strain.name: [] for strain in strains}
    for row in rows:
        by_name[row["strain"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharex=True)
    for strain in strains:
        strain_rows = sorted(by_name[strain.name], key=lambda r: r["T_w"])
        t_ws = [r["T_w"] for r in strain_rows]
        medians = [r["median"] for r in strain_rows]
        p90s = [r["p90"] for r in strain_rows]
        label = "ancestor" if strain.kind == "ancestor" else strain.name
        lw = 2.5 if strain.kind == "ancestor" else 1.6
        axes[0].plot(t_ws, medians, marker="o", lw=lw, label=label)
        axes[1].plot(t_ws, p90s, marker="o", lw=lw, label=label)
        if train_t_w in t_ws:
            train_i = t_ws.index(train_t_w)
            axes[0].plot(
                [train_t_w],
                [medians[train_i]],
                marker="*",
                markersize=12,
                color=axes[0].lines[-1].get_color(),
                markeredgecolor="black",
                markeredgewidth=0.55,
            )
            axes[1].plot(
                [train_t_w],
                [p90s[train_i]],
                marker="*",
                markersize=12,
                color=axes[1].lines[-1].get_color(),
                markeredgecolor="black",
                markeredgewidth=0.55,
            )

    axes[0].set_title("Median lag")
    axes[1].set_title("90th percentile lag")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("T_w")
        ax.set_ylabel("Recovery lag")
        ax.axvline(
            train_t_w,
            color="0.2",
            ls="--",
            lw=1.1,
            alpha=0.75,
            label="train T_w" if ax is axes[1] else None,
        )
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle("RCCN lag versus waiting time")
    fig.tight_layout()
    fig.savefig(out_dir / "lag_vs_Tw_median_p90.png", dpi=220)
    fig.savefig(out_dir / "lag_vs_Tw_median_p90.pdf")
    plt.close(fig)


def plot_aging_cdf_grid(
    lags_by_strain_tw: dict[str, dict[int, np.ndarray]],
    strains: list[Strain],
    out_dir: Path,
) -> None:
    cmap = colors.LinearSegmentedColormap(
        "rccn_tw",
        {
            "red": ((0.0, 0.22, 0.0), (0.5, 1.0, 1.0), (1.0, 0.89, 1.0)),
            "green": ((0.0, 0.49, 0.0), (0.5, 1.0, 1.0), (1.0, 0.12, 1.0)),
            "blue": ((0.0, 0.72, 0.0), (0.5, 0.0, 0.0), (1.0, 0.11, 1.0)),
        },
    )
    max_lag = max(
        float(np.max(lags))
        for by_tw in lags_by_strain_tw.values()
        for lags in by_tw.values()
    )
    n_cols = min(3, len(strains))
    n_rows = int(np.ceil(len(strains) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.1 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for ax, strain in zip(axes_arr, strains):
        by_tw = lags_by_strain_tw[strain.name]
        t_ws = sorted(by_tw)
        for color_i, t_w in zip(np.linspace(0.25, 1.0, len(t_ws)), t_ws):
            xs, ys = one_minus_cdf(by_tw[t_w])
            ax.plot(xs, ys, color=cmap(color_i), lw=1.8, label=f"T_w={t_w}")
        ax.set_title(strain.name)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlim(0, max(50.0, max_lag * 1.05))
        ax.set_xlabel("Recovery lag")
        ax.set_ylabel("1-CDF")
        ax.grid(alpha=0.22)

    for ax in axes_arr[len(strains) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle("Lag 1-CDFs across waiting times, 2000 realizations", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "aging_1cdf_grid_2000.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "aging_1cdf_grid_2000.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_individual_aging_cdfs(
    lags_by_strain_tw: dict[str, dict[int, np.ndarray]],
    strains: list[Strain],
    out_dir: Path,
) -> None:
    cmap = colors.LinearSegmentedColormap(
        "rccn_tw_single",
        {
            "red": ((0.0, 0.22, 0.0), (0.5, 1.0, 1.0), (1.0, 0.89, 1.0)),
            "green": ((0.0, 0.49, 0.0), (0.5, 1.0, 1.0), (1.0, 0.12, 1.0)),
            "blue": ((0.0, 0.72, 0.0), (0.5, 0.0, 0.0), (1.0, 0.11, 1.0)),
        },
    )
    for strain in strains:
        by_tw = lags_by_strain_tw[strain.name]
        t_ws = sorted(by_tw)
        max_lag = max(float(np.max(lags)) for lags in by_tw.values())
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for color_i, t_w in zip(np.linspace(0.25, 1.0, len(t_ws)), t_ws):
            xs, ys = one_minus_cdf(by_tw[t_w])
            ax.plot(xs, ys, color=cmap(color_i), lw=2.0, label=f"T_w={t_w}")
        ax.set_title(f"{strain.name}: lag 1-CDFs across waiting times")
        ax.set_xlabel("Recovery lag")
        ax.set_ylabel("1-CDF")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlim(0, max(50.0, max_lag * 1.05))
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"aging_1cdf_{strain.name}_2000.png", dpi=240)
        fig.savefig(out_dir / f"aging_1cdf_{strain.name}_2000.pdf")
        plt.close(fig)


def plot_tail_cdfs(
    tail_lags: dict[str, np.ndarray],
    strains: list[Strain],
    tail_t_w: int,
    tail_realizations: int,
    out_dir: Path,
) -> None:
    y_min = min(1e-5, 0.5 / tail_realizations)
    ancestor = strains[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for strain in strains:
        xs, ys = one_minus_cdf(tail_lags[strain.name])
        lw = 2.5 if strain.kind == "ancestor" else 1.5
        alpha = 1.0 if strain.kind == "ancestor" else 0.86
        ax.plot(xs, ys, lw=lw, alpha=alpha, label=strain.name)
    ax.set_title(f"Deep-tail lag 1-CDF at T_w={tail_t_w}, n={tail_realizations}")
    ax.set_xlabel("Recovery lag")
    ax.set_ylabel("1-CDF")
    ax.set_yscale("log")
    ax.set_ylim(y_min, 1.1)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"tail_1cdf_Tw_{tail_t_w}_n_{tail_realizations}.png", dpi=240)
    fig.savefig(out_dir / f"tail_1cdf_Tw_{tail_t_w}_n_{tail_realizations}.pdf")
    plt.close(fig)

    for strain in strains[1:]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for plotted in (ancestor, strain):
            xs, ys = one_minus_cdf(tail_lags[plotted.name])
            ax.plot(xs, ys, lw=2.2, label=plotted.name)
        ax.set_title(f"{ancestor.name} vs {strain.name}, T_w={tail_t_w}")
        ax.set_xlabel("Recovery lag")
        ax.set_ylabel("1-CDF")
        ax.set_yscale("log")
        ax.set_ylim(y_min, 1.1)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"tail_1cdf_{ancestor.name}_vs_{strain.name}_Tw_{tail_t_w}.png",
            dpi=240,
        )
        fig.savefig(out_dir / f"tail_1cdf_{ancestor.name}_vs_{strain.name}_Tw_{tail_t_w}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ancestor-j",
        type=Path,
        default=Path("outputs/ancestor_aging_additive_update/J_matrices/ancestor_J_00.npy"),
    )
    parser.add_argument("--topology", type=Path, default=Path("data/topology_1.npy"))
    parser.add_argument("--evolved", action="append", type=parse_strain, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evolved_lag_characterization"))
    parser.add_argument("--t-ws", default="20,63,200,632,2000")
    parser.add_argument("--n-realizations", type=int, default=2000)
    parser.add_argument("--tail-t-w", type=int, default=200)
    parser.add_argument("--train-t-w", type=int, default=200)
    parser.add_argument("--tail-realizations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--equilibration-time", type=int, default=4000)
    parser.add_argument("--relaxation-time", type=int, default=4000)
    args = parser.parse_args()

    out_dir = args.out_dir
    lag_dir = out_dir / "lag_times"
    out_dir.mkdir(parents=True, exist_ok=True)
    lag_dir.mkdir(exist_ok=True)

    strains = [
        Strain(name="ancestor_J0", j_path=args.ancestor_j, kind="ancestor"),
        *args.evolved,
    ]
    t_ws = parse_t_ws(args.t_ws)
    topology = np.load(args.topology)

    metadata = {
        "ancestor_j": str(args.ancestor_j),
        "topology": str(args.topology),
        "strains": [strain.__dict__ | {"j_path": str(strain.j_path)} for strain in strains],
        "t_ws": t_ws,
        "n_realizations": args.n_realizations,
        "tail_t_w": args.tail_t_w,
        "train_t_w": args.train_t_w,
        "tail_realizations": args.tail_realizations,
        "seed": args.seed,
        "backend": args.backend,
        "device": args.device,
        "equilibration_time": args.equilibration_time,
        "relaxation_time": args.relaxation_time,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    summary_rows: list[dict] = []
    lags_by_strain_tw: dict[str, dict[int, np.ndarray]] = {}
    Js = {strain.name: np.load(strain.j_path) for strain in strains}

    for strain in strains:
        lags_by_strain_tw[strain.name] = {}
        for t_w in t_ws:
            lags = load_or_simulate(
                J=Js[strain.name],
                topology=topology,
                strain=strain,
                t_w=t_w,
                n_realizations=args.n_realizations,
                lag_dir=lag_dir,
                seed=args.seed,
                backend=args.backend,
                device=args.device,
                equilibration_time=args.equilibration_time,
                relaxation_time=args.relaxation_time,
            )
            lags_by_strain_tw[strain.name][t_w] = lags
            summary_rows.append(
                {
                    "strain": strain.name,
                    "kind": strain.kind,
                    "T_w": t_w,
                    "n_realizations": args.n_realizations,
                    **finite_summary(lags),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "tw_done",
                        "strain": strain.name,
                        "T_w": t_w,
                        "n_realizations": args.n_realizations,
                        **finite_summary(lags),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    tail_lags: dict[str, np.ndarray] = {}
    tail_rows: list[dict] = []
    for strain in strains:
        lags = load_or_simulate(
            J=Js[strain.name],
            topology=topology,
            strain=strain,
            t_w=args.tail_t_w,
            n_realizations=args.tail_realizations,
            lag_dir=lag_dir,
            seed=args.seed,
            backend=args.backend,
            device=args.device,
            equilibration_time=args.equilibration_time,
            relaxation_time=args.relaxation_time,
        )
        tail_lags[strain.name] = lags
        tail_rows.append(
            {
                "strain": strain.name,
                "kind": strain.kind,
                "T_w": args.tail_t_w,
                "n_realizations": args.tail_realizations,
                **finite_summary(lags),
            }
        )
        print(
            json.dumps(
                {
                    "event": "tail_done",
                    "strain": strain.name,
                    "T_w": args.tail_t_w,
                    "n_realizations": args.tail_realizations,
                    **finite_summary(lags),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    write_csv(out_dir / "lag_vs_Tw_summary_2000.csv", summary_rows)
    write_csv(out_dir / f"tail_summary_Tw_{args.tail_t_w}_n_{args.tail_realizations}.csv", tail_rows)
    plot_lag_vs_tw(summary_rows, strains, out_dir, args.train_t_w)
    plot_aging_cdf_grid(lags_by_strain_tw, strains, out_dir)
    plot_individual_aging_cdfs(lags_by_strain_tw, strains, out_dir)
    plot_tail_cdfs(tail_lags, strains, args.tail_t_w, args.tail_realizations, out_dir)

    print(json.dumps({"event": "done", "out_dir": str(out_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
