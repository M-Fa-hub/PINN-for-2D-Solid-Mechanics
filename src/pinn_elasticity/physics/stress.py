"""Stress tensor utilities including von Mises equivalent stress."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch


class Stress(NamedTuple):
    sigma_xx: torch.Tensor
    sigma_yy: torch.Tensor
    sigma_xy: torch.Tensor


def von_mises(
    stress: Stress,
    plane_condition: Literal["plane_stress", "plane_strain"] = "plane_stress",
) -> torch.Tensor:
    """
    von Mises equivalent stress.

    Plane stress (σ_zz = 0):
        σ_vm = sqrt(σ_xx² - σ_xx σ_yy + σ_yy² + 3 σ_xy²)

    Plane strain (σ_zz = ν(σ_xx + σ_yy) is not used here directly;
    for evaluation we use the in-plane formula with σ_zz = 0 unless
    poisson is provided — see von_mises_plane_strain).
    """
    sxx, syy, sxy = stress.sigma_xx, stress.sigma_yy, stress.sigma_xy
    if plane_condition == "plane_stress":
        return torch.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
    # Default in-plane expression; prefer von_mises_plane_strain when ν known
    return torch.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def von_mises_plane_strain(stress: Stress, nu: float | torch.Tensor) -> torch.Tensor:
    """von Mises with σ_zz = ν(σ_xx + σ_yy) for plane strain."""
    sxx, syy, sxy = stress.sigma_xx, stress.sigma_yy, stress.sigma_xy
    szz = nu * (sxx + syy)
    return torch.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy**2
    )


def traction(stress: Stress, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Surface traction t = σ · n.

    normal: (2,) or (N, 2)
    Returns (t_x, t_y) each (N, 1).
    """
    if normal.dim() == 1:
        nx, ny = normal[0], normal[1]
        tx = stress.sigma_xx * nx + stress.sigma_xy * ny
        ty = stress.sigma_xy * nx + stress.sigma_yy * ny
    else:
        nx = normal[:, 0:1]
        ny = normal[:, 1:2]
        tx = stress.sigma_xx * nx + stress.sigma_xy * ny
        ty = stress.sigma_xy * nx + stress.sigma_yy * ny
    return tx, ty
