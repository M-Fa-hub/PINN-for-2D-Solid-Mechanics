"""Equilibrium residuals for static linear elasticity."""

from __future__ import annotations

from typing import NamedTuple

import torch

from pinn_elasticity.physics.stress import Stress


class EquilibriumResidual(NamedTuple):
    r_x: torch.Tensor
    r_y: torch.Tensor

    def magnitude(self) -> torch.Tensor:
        return torch.sqrt(self.r_x**2 + self.r_y**2)


def compute_stress_divergence(
    stress: Stress,
    xy: torch.Tensor,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y) via autograd.
    """
    sxx, syy, sxy = stress.sigma_xx, stress.sigma_yy, stress.sigma_xy
    ones_xx = torch.ones_like(sxx)
    ones_yy = torch.ones_like(syy)
    ones_xy = torch.ones_like(sxy)

    grad_sxx = torch.autograd.grad(
        sxx, xy, grad_outputs=ones_xx, create_graph=create_graph, retain_graph=True
    )[0]
    grad_syy = torch.autograd.grad(
        syy, xy, grad_outputs=ones_yy, create_graph=create_graph, retain_graph=True
    )[0]
    grad_sxy = torch.autograd.grad(
        sxy, xy, grad_outputs=ones_xy, create_graph=create_graph, retain_graph=True
    )[0]

    div_x = grad_sxx[:, 0:1] + grad_sxy[:, 1:2]
    div_y = grad_sxy[:, 0:1] + grad_syy[:, 1:2]
    return div_x, div_y


def compute_equilibrium_residual(
    stress: Stress,
    xy: torch.Tensor,
    body_force: tuple[float, float] | tuple[torch.Tensor, torch.Tensor] = (0.0, 0.0),
    create_graph: bool = True,
) -> EquilibriumResidual:
    """
    Static equilibrium residual:

        r_x = ∂σ_xx/∂x + ∂σ_xy/∂y + f_x
        r_y = ∂σ_xy/∂x + ∂σ_yy/∂y + f_y
    """
    div_x, div_y = compute_stress_divergence(stress, xy, create_graph=create_graph)
    fx, fy = body_force
    if not torch.is_tensor(fx):
        fx = torch.as_tensor(fx, dtype=xy.dtype, device=xy.device)
        fy = torch.as_tensor(fy, dtype=xy.dtype, device=xy.device)
    return EquilibriumResidual(r_x=div_x + fx, r_y=div_y + fy)
