from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh_tridiagonal

ORBITAL_LETTERS = "spdfghijklm"
EV_PER_HARTREE = 27.211386245988


@dataclass(frozen=True)
class GridSpec:
    r_min: float
    r_max: float
    n_points: int


@dataclass(frozen=True)
class Nucleus:
    z: int
    n: int

    @property
    def a(self) -> int:
        return self.z + self.n


@dataclass(frozen=True)
class WoodsSaxonParams:
    r0_fm: float = 1.27
    a_fm: float = 0.67
    v0_mev: float = -51.0
    asym_mev: float = 33.0
    so_scale: float = -0.44


def radial_bound_energies(
    potential_fn: Callable[[np.ndarray], np.ndarray],
    l: int,
    mass_coeff: float,
    grid: GridSpec,
    e_min: float,
    e_max: float = 0.0,
) -> np.ndarray:
    r = np.linspace(grid.r_min, grid.r_max, grid.n_points)
    dr = r[1] - r[0]

    r_in = r[1:-1]
    kin_diag = 2.0 * mass_coeff / dr**2
    kin_off = -mass_coeff / dr**2
    cent = mass_coeff * l * (l + 1) / (r_in**2)
    v_eff = potential_fn(r_in) + cent

    diag = kin_diag + v_eff
    off = np.full(r_in.size - 1, kin_off)

    evals = eigh_tridiagonal(
        diag,
        off,
        select="v",
        select_range=(e_min, e_max),
        check_finite=False,
    )[0]
    return evals


def ws_form_factor(r: np.ndarray, r_radius: float, diffuseness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((r - r_radius) / diffuseness))


def ws_dfdr(r: np.ndarray, r_radius: float, diffuseness: float) -> np.ndarray:
    f = ws_form_factor(r, r_radius, diffuseness)
    return -(f * (1.0 - f)) / diffuseness


def ls_expectation(l: int, j: float) -> float:
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def ws_depth_mev(nucleus: Nucleus, species: str, p: WoodsSaxonParams) -> float:
    asym = (nucleus.n - nucleus.z) / nucleus.a
    if species == "neutron":
        return p.v0_mev + p.asym_mev * asym
    if species == "proton":
        return p.v0_mev - p.asym_mev * asym
    raise ValueError(f"Unknown species: {species}")


def ws_potential(
    r: np.ndarray,
    nucleus: Nucleus,
    species: str,
    l: int,
    j: float,
    p: WoodsSaxonParams,
) -> np.ndarray:
    r_radius = p.r0_fm * nucleus.a ** (1.0 / 3.0)
    v_depth = ws_depth_mev(nucleus, species, p)
    v_ls = p.so_scale * v_depth

    f = ws_form_factor(r, r_radius, p.a_fm)
    central = v_depth * f

    so_eig = ls_expectation(l, j)
    so = v_ls * so_eig * (p.r0_fm**2) * (ws_dfdr(r, r_radius, p.a_fm) / r)

    return central + so


def spectroscopic_label(n_radial_index: int, l: int, j: float) -> str:
    n_shell = n_radial_index + 1
    letter = ORBITAL_LETTERS[l] if l < len(ORBITAL_LETTERS) else f"l{l}"
    j2 = int(round(2 * j))
    return f"{n_shell}{letter}{j2}/2"


def bound_levels_ws(
    nucleus: Nucleus,
    species: str,
    grid: GridSpec,
    ws_params: WoodsSaxonParams,
    l_max: int = 7,
    e_window: Tuple[float, float] = (-120.0, 0.0),
) -> List[Tuple[str, float, int]]:
    hbar2_over_2m = 20.73553  # MeV fm^2
    levels: List[Tuple[str, float, int]] = []

    for l in range(l_max + 1):
        js: Sequence[float] = (l + 0.5,) if l == 0 else (l - 0.5, l + 0.5)
        for j in js:
            pot = lambda r, l=l, j=j: ws_potential(r, nucleus, species, l, j, ws_params)
            evals = radial_bound_energies(
                potential_fn=pot,
                l=l,
                mass_coeff=hbar2_over_2m,
                grid=grid,
                e_min=e_window[0],
                e_max=e_window[1],
            )
            for nr, e in enumerate(evals):
                levels.append((spectroscopic_label(nr, l, j), float(e), int(2 * j + 1)))

    levels.sort(key=lambda x: x[1])
    return levels


def hydrogen_lowest_three(grid: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    vals_all: List[float] = []
    for l in range(0, 5):
        evals = radial_bound_energies(
            potential_fn=lambda r: -1.0 / r,
            l=l,
            mass_coeff=0.5,
            grid=grid,
            e_min=-2.0,
            e_max=0.0,
        )
        vals_all.extend(evals.tolist())

    vals = np.array(sorted(vals_all))
    unique_vals = [vals[0]]
    for x in vals[1:]:
        if abs(x - unique_vals[-1]) > 2e-3:
            unique_vals.append(x)
        if len(unique_vals) >= 3:
            break

    return np.array(unique_vals[:3]), np.array([-0.5, -0.125, -1.0 / 18.0])


def neon_potential(r: np.ndarray, rc: float, z_inner: float = 10.0) -> np.ndarray:
    z_eff = 1.0 + (z_inner - 1.0) * np.exp(-(r / rc) ** 2)
    return -z_eff / r


def neon_2p_energy(rc: float, grid: GridSpec) -> float:
    evals = radial_bound_energies(
        potential_fn=lambda r, rc=rc: neon_potential(r, rc),
        l=1,
        mass_coeff=0.5,
        grid=grid,
        e_min=-20.0,
        e_max=0.0,
    )
    if evals.size == 0:
        raise RuntimeError("No bound p-states found for this rc")
    return float(evals[0])


def tune_neon_rc(grid: GridSpec, target_e_ha: float) -> Tuple[float, float]:
    samples = np.linspace(0.15, 4.0, 60)
    fvals = [neon_2p_energy(float(rc), grid) - target_e_ha for rc in samples]

    left = right = None
    for i in range(len(samples) - 1):
        if fvals[i] == 0.0:
            rc = float(samples[i])
            return rc, neon_2p_energy(rc, grid)
        if fvals[i] * fvals[i + 1] < 0.0:
            left, right = float(samples[i]), float(samples[i + 1])
            break

    if left is None or right is None:
        rc = float(samples[int(np.argmin(np.abs(fvals)))])
        return rc, neon_2p_energy(rc, grid)

    for _ in range(50):
        mid = 0.5 * (left + right)
        f_left = neon_2p_energy(left, grid) - target_e_ha
        f_mid = neon_2p_energy(mid, grid) - target_e_ha
        if abs(f_mid) < 1e-6:
            return mid, neon_2p_energy(mid, grid)
        if f_left * f_mid < 0.0:
            right = mid
        else:
            left = mid

    rc = 0.5 * (left + right)
    return rc, neon_2p_energy(rc, grid)
