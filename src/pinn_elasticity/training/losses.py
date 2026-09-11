"""Training loss assembly for forward and inverse PINNs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pinn_elasticity.boundary.conditions import BoundaryConditionSet
from pinn_elasticity.boundary.losses import boundary_loss
from pinn_elasticity.config import LossConfig
from pinn_elasticity.physics.constitutive import compute_stress
from pinn_elasticity.physics.equilibrium import compute_equilibrium_residual
from pinn_elasticity.physics.kinematics import (
    compute_displacement_gradients,
    compute_strain,
    scale_strain,
)
from pinn_elasticity.scaling import ScaleFactors


@dataclass
class LossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    bc: torch.Tensor
    dirichlet: torch.Tensor
    neumann: torch.Tensor
    data: torch.Tensor


def pde_loss(
    model: nn.Module,
    xy_interior: torch.Tensor,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    scales: ScaleFactors,
    plane_condition: str = "plane_stress",
    body_force: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """
    Mean squared equilibrium residual.

    Residual is ∇_* · σ_phys + f_phys * L_ref (see ScaleFactors docstring).
    """
    xy = xy_interior.clone().requires_grad_(True)
    uv = model(xy)
    grads = compute_displacement_gradients(uv, xy, create_graph=True)
    strain = scale_strain(compute_strain(grads), scales.strain_scale)
    stress = compute_stress(strain, E, nu, plane_condition=plane_condition)  # type: ignore[arg-type]
    fx, fy = body_force
    L = scales.length_scale
    resid = compute_equilibrium_residual(
        stress, xy, body_force=(fx * L, fy * L), create_graph=True
    )
    return torch.mean(resid.r_x**2 + resid.r_y**2)


def data_loss(
    model: nn.Module,
    xy_data: torch.Tensor | None,
    uv_data: torch.Tensor | None,
) -> torch.Tensor:
    """Optional MSE against sparse displacement observations (network units)."""
    if xy_data is None or uv_data is None or xy_data.numel() == 0:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return torch.tensor(0.0, device=device, dtype=dtype)
    pred = model(xy_data)
    return torch.mean((pred - uv_data) ** 2)


def compute_total_loss(
    model: nn.Module,
    xy_interior: torch.Tensor,
    bcs: BoundaryConditionSet,
    loss_cfg: LossConfig,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    scales: ScaleFactors,
    plane_condition: str = "plane_stress",
    body_force: tuple[float, float] = (0.0, 0.0),
    soft_dirichlet: bool = True,
    xy_data: torch.Tensor | None = None,
    uv_data: torch.Tensor | None = None,
) -> LossBreakdown:
    """Assemble weighted total loss."""
    L_pde = pde_loss(
        model,
        xy_interior,
        E,
        nu,
        scales=scales,
        plane_condition=plane_condition,
        body_force=body_force,
    )
    L_bc, L_d, L_n = boundary_loss(
        model,
        bcs,
        E,
        nu,
        scales=scales,
        plane_condition=plane_condition,
        soft_dirichlet=soft_dirichlet,
        lambda_dirichlet=loss_cfg.lambda_dirichlet,
        lambda_neumann=loss_cfg.lambda_neumann,
    )
    L_data = data_loss(model, xy_data, uv_data)
    total = (
        loss_cfg.lambda_pde * L_pde
        + loss_cfg.lambda_bc * L_bc
        + loss_cfg.lambda_data * L_data
    )
    return LossBreakdown(
        total=total,
        pde=L_pde,
        bc=L_bc,
        dirichlet=L_d,
        neumann=L_n,
        data=L_data,
    )
