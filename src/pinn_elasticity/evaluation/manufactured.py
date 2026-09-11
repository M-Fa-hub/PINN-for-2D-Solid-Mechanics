"""Manufactured quadratic solution helpers for verification and training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pinn_elasticity.physics.constitutive import compute_stress
from pinn_elasticity.physics.kinematics import Strain


@dataclass(frozen=True)
class ManufacturedField:
    """
    u = A x² + B y²
    v = C x² + D y²
    """

    A: float = 1e-3
    B: float = 2e-3
    C: float = -1.5e-3
    D: float = 0.5e-3

    def displacement(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        u = self.A * x**2 + self.B * y**2
        v = self.C * x**2 + self.D * y**2
        return torch.cat([u, v], dim=1)

    def body_force(
        self,
        E: float,
        nu: float,
        plane_condition: str = "plane_stress",
    ) -> tuple[float, float]:
        """Constant body force making the manufactured field an exact solution."""
        # ε_xx = 2 A x, ε_yy = 2 D y, γ_xy = 2 B y + 2 C x
        # div σ is constant for this quadratic field under Hooke
        if plane_condition == "plane_stress":
            factor = E / (1.0 - nu**2)
            G = E / (2.0 * (1.0 + nu))
            div_x = factor * 2.0 * self.A + G * 2.0 * self.B
            div_y = G * 2.0 * self.C + factor * 2.0 * self.D
        else:
            from pinn_elasticity.physics.constitutive import lame_from_E_nu

            lam, mu = lame_from_E_nu(E, nu)
            div_x = (lam + 2.0 * mu) * 2.0 * self.A + mu * 2.0 * self.B
            div_y = mu * 2.0 * self.C + (lam + 2.0 * mu) * 2.0 * self.D
        return -div_x, -div_y

    def stress_at(
        self,
        xy: torch.Tensor,
        E: float,
        nu: float,
        plane_condition: str = "plane_stress",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        strain = Strain(
            eps_xx=2.0 * self.A * x,
            eps_yy=2.0 * self.D * y,
            gamma_xy=2.0 * self.B * y + 2.0 * self.C * x,
        )
        s = compute_stress(strain, E, nu, plane_condition=plane_condition)  # type: ignore[arg-type]
        return s.sigma_xx, s.sigma_yy, s.sigma_xy
