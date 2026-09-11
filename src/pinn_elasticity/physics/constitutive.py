"""Linear elastic constitutive relations for plane stress / plane strain."""

from __future__ import annotations

from typing import Literal

import torch

from pinn_elasticity.physics.kinematics import Strain
from pinn_elasticity.physics.stress import Stress


def lame_from_E_nu(E: float | torch.Tensor, nu: float | torch.Tensor) -> tuple:
    """Return (λ, μ) from E, ν (3D isotropic). Useful for plane strain."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def compute_stress(
    strain: Strain,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    plane_condition: Literal["plane_stress", "plane_strain"] = "plane_stress",
) -> Stress:
    """
    Hooke's law for 2D isotropic linear elasticity.

    Plane stress:
        σ_xx = E/(1-ν²) (ε_xx + ν ε_yy)
        σ_yy = E/(1-ν²) (ε_yy + ν ε_xx)
        σ_xy = E/(2(1+ν)) γ_xy

    Plane strain:
        σ_xx = (λ+2μ) ε_xx + λ ε_yy
        σ_yy = (λ+2μ) ε_yy + λ ε_xx
        σ_xy = μ γ_xy
    """
    eps_xx, eps_yy, gamma_xy = strain.eps_xx, strain.eps_yy, strain.gamma_xy

    if plane_condition == "plane_stress":
        factor = E / (1.0 - nu**2)
        sigma_xx = factor * (eps_xx + nu * eps_yy)
        sigma_yy = factor * (eps_yy + nu * eps_xx)
        sigma_xy = (E / (2.0 * (1.0 + nu))) * gamma_xy
    elif plane_condition == "plane_strain":
        lam, mu = lame_from_E_nu(E, nu)
        sigma_xx = (lam + 2.0 * mu) * eps_xx + lam * eps_yy
        sigma_yy = (lam + 2.0 * mu) * eps_yy + lam * eps_xx
        sigma_xy = mu * gamma_xy
    else:
        raise ValueError(f"Unknown plane_condition: {plane_condition}")

    return Stress(sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_xy=sigma_xy)
