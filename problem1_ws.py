from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from common_functions import GridSpec, Nucleus, WoodsSaxonParams, bound_levels_ws


def print_levels(levels: List[Tuple[str, float, int]], species: str) -> None:
    print(f"\n  {species.capitalize()} bound levels (E < 0):")
    if not levels:
        print("    none")
        return
    for label, energy, deg in levels:
        print(f"    {label:7s}  E = {energy:8.3f} MeV   (2j+1={deg})")


def plot_levels(results: Dict[str, Dict[str, List[Tuple[str, float, int]]]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    for ax, nucleus_key in zip(axes, ["Z20N20", "Z20N28"]):
        neutron_levels = results[nucleus_key]["neutron"]
        proton_levels = results[nucleus_key]["proton"]

        for label, energy, _ in neutron_levels:
            ax.hlines(energy, -0.35, -0.05, color="#1f77b4", linewidth=2)
            ax.text(-0.38, energy, label, fontsize=8, ha="right", va="center", color="#1f77b4")

        for label, energy, _ in proton_levels:
            ax.hlines(energy, 0.05, 0.35, color="#d62728", linewidth=2)
            ax.text(0.38, energy, label, fontsize=8, ha="left", va="center", color="#d62728")

        ax.set_xlim(-0.5, 0.65)
        ax.set_xticks([-0.2, 0.2])
        ax.set_xticklabels(["n", "p"])
        ax.set_title(nucleus_key)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Energy (MeV)")
    fig.suptitle("Problem 1: Woods-Saxon + Spin-Orbit Bound Levels")
    fig.tight_layout()

    out_path = out_dir / "problem1_ws_levels.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 80)
    print("Problem 1: Woods-Saxon + spin-orbit bound eigenvalues (MeV)")
    print("=" * 80)

    ws_params = WoodsSaxonParams()
    grid = GridSpec(r_min=1e-4, r_max=20.0, n_points=1200)

    nuclei = {
        "Z20N20": Nucleus(z=20, n=20),
        "Z20N28": Nucleus(z=20, n=28),
    }

    results: Dict[str, Dict[str, List[Tuple[str, float, int]]]] = {}

    for key, nuc in nuclei.items():
        print(f"\nNucleus: Z={nuc.z}, N={nuc.n}, A={nuc.a}")
        results[key] = {}
        for species in ("neutron", "proton"):
            levels = bound_levels_ws(nuc, species, grid, ws_params)
            results[key][species] = levels
            print_levels(levels, species)

    out_dir = Path(__file__).resolve().parent
    out_path = plot_levels(results, out_dir)
    print(f"\nSaved plot: {out_path.name}")


if __name__ == "__main__":
    main()
