from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_functions import (
    EV_PER_HARTREE,
    GridSpec,
    hydrogen_lowest_three,
    neon_potential,
    tune_neon_rc,
)


def plot_hydrogen_comparison(numeric: np.ndarray, exact: np.ndarray, out_dir: Path) -> Path:
    nvals = np.array([1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nvals, exact, "o-", label="Exact", color="#2ca02c", linewidth=2)
    ax.plot(nvals, numeric, "s--", label="Numeric", color="#1f77b4", linewidth=2)
    ax.set_xlabel("Principal quantum number n")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("Problem 2: Hydrogen Lowest 3 Energies")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "problem2_hydrogen_check.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_neon_potential(rc: float, e2p_ha: float, out_dir: Path) -> Path:
    r = np.linspace(0.05, 8.0, 600)
    v_ne = neon_potential(r, rc)
    v_h = -1.0 / r

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, v_ne, label="Ne interpolated", color="#d62728", linewidth=2)
    ax.plot(r, v_h, label="Hydrogenic -1/r", color="#1f77b4", linewidth=1.5, alpha=0.8)
    ax.axhline(e2p_ha, color="black", linestyle=":", linewidth=1.5, label="Ne 2p energy")
    ax.set_ylim(-12, 0.5)
    ax.set_xlabel("r (a0)")
    ax.set_ylabel("V(r) (Hartree)")
    ax.set_title("Problem 2: Neon Interpolated Potential")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "problem2_neon_potential.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 80)
    print("Problem 2: Hydrogen check + Neon interpolation")
    print("=" * 80)

    grid = GridSpec(r_min=1e-4, r_max=120.0, n_points=2500)

    numeric, exact = hydrogen_lowest_three(grid)
    print("\nHydrogen lowest 3 energies (Hartree):")
    for i, (num, ex) in enumerate(zip(numeric, exact), start=1):
        print(f"  n={i}: numeric={num: .8f}, exact={ex: .8f}, error={num - ex: .3e}")

    target_ev = 21.5645
    target_ha = -target_ev / EV_PER_HARTREE
    rc, e2p = tune_neon_rc(grid, target_ha)

    print("\nNeon interpolated potential: Zeff(r)=1+9*exp(-(r/rc)^2)")
    print(f"  tuned rc = {rc:.6f} a0")
    print(f"  target E_2p = {target_ha:.8f} Ha  ({-target_ha * EV_PER_HARTREE:.4f} eV)")
    print(f"  model  E_2p = {e2p:.8f} Ha  ({-e2p * EV_PER_HARTREE:.4f} eV)")

    out_dir = Path(__file__).resolve().parent
    p1 = plot_hydrogen_comparison(numeric, exact, out_dir)
    p2 = plot_neon_potential(rc, e2p, out_dir)

    print(f"\nSaved plot: {p1.name}")
    print(f"Saved plot: {p2.name}")


if __name__ == "__main__":
    main()
