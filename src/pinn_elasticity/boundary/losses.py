"""Boundary-condition loss terms (soft enforcement)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn_elasticity.boundary.conditions import BoundaryConditionSet, DirichletBC, NeumannBC
from pinn_elasticity.physics.constitutive import compute_stress
from pinn_elasticity.physics.kinematics import (
    compute_displacement_gradients,
    compute_strain,
    scale_strain,
)
from pinn_elasticity.physics.stress import traction
from pinn_elasticity.scaling import ScaleFactors


def _as_tensor(val: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(val):
        out = val.to(dtype=like.dtype, device=like.device)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[0] == 1 and like.shape[0] > 1:
            out = out.expand(like.shape[0], -1)
        return out
    return torch.full((like.shape[0], 1), float(val), dtype=like.dtype, device=like.device)


def dirichlet_loss(
    model: nn.Module,
    bc: DirichletBC,
    soft: bool = True,
) -> torch.Tensor:
    """MSE between predicted and prescribed displacements (network / nondim units)."""
    if not soft:
        return torch.tensor(0.0, device=bc.segment.points.device)
    xy = bc.segment.points.requires_grad_(False)
    uv = model(xy)
    u_t = _as_tensor(bc.u, uv)
    v_t = _as_tensor(bc.v, uv)
    return torch.mean((uv[:, 0:1] - u_t) ** 2 + (uv[:, 1:2] - v_t) ** 2)


def neumann_loss(
    model: nn.Module,
    bc: NeumannBC,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    scales: ScaleFactors,
    plane_condition: str = "plane_stress",
) -> torch.Tensor:
    """MSE between predicted physical traction σ·n and prescribed physical traction."""
    xy = bc.segment.points.clone().requires_grad_(True)
    uv = model(xy)
    grads = compute_displacement_gradients(uv, xy, create_graph=True)
    strain = scale_strain(compute_strain(grads), scales.strain_scale)
    stress = compute_stress(strain, E, nu, plane_condition=plane_condition)  # type: ignore[arg-type]
    n = bc.segment.normal.to(dtype=xy.dtype, device=xy.device)
    tx_pred, ty_pred = traction(stress, n)
    tx_t = _as_tensor(bc.tx, tx_pred)
    ty_t = _as_tensor(bc.ty, ty_pred)
    return torch.mean((tx_pred - tx_t) ** 2 + (ty_pred - ty_t) ** 2)


def boundary_loss(
    model: nn.Module,
    bcs: BoundaryConditionSet,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    scales: ScaleFactors,
    plane_condition: str = "plane_stress",
    soft_dirichlet: bool = True,
    lambda_dirichlet: float = 1.0,
    lambda_neumann: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (L_bc_total, L_dirichlet, L_neumann)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    L_d = torch.tensor(0.0, device=device, dtype=dtype)
    for bc in bcs.dirichlet:
        L_d = L_d + dirichlet_loss(model, bc, soft=soft_dirichlet)
    L_n = torch.tensor(0.0, device=device, dtype=dtype)
    for nbc in bcs.neumann:
        L_n = L_n + neumann_loss(
            model, nbc, E, nu, scales=scales, plane_condition=plane_condition
        )
    L_bc = lambda_dirichlet * L_d + lambda_neumann * L_n
    return L_bc, L_d, L_n
