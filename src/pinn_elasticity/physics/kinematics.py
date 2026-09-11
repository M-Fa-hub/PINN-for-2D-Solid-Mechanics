"""Kinematics: displacement gradients and infinitesimal strain."""

from __future__ import annotations

from typing import NamedTuple

import torch


class DisplacementGradients(NamedTuple):
    du_dx: torch.Tensor
    du_dy: torch.Tensor
    dv_dx: torch.Tensor
    dv_dy: torch.Tensor


class Strain(NamedTuple):
    """Infinitesimal strain components (engineering shear γ_xy)."""

    eps_xx: torch.Tensor
    eps_yy: torch.Tensor
    gamma_xy: torch.Tensor


def compute_displacement_gradients(
    uv: torch.Tensor,
    xy: torch.Tensor,
    create_graph: bool = True,
) -> DisplacementGradients:
    """
    Compute ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y via reverse-mode AD.

    Parameters
    ----------
    uv : (N, 2) displacements requiring grad w.r.t. xy
    xy : (N, 2) coordinates with requires_grad=True
    """
    if xy.shape != uv.shape:
        raise ValueError(f"xy and uv shape mismatch: {xy.shape} vs {uv.shape}")
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    grad_u = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u), create_graph=create_graph, retain_graph=True
    )[0]
    grad_v = torch.autograd.grad(
        v, xy, grad_outputs=torch.ones_like(v), create_graph=create_graph, retain_graph=True
    )[0]
    return DisplacementGradients(
        du_dx=grad_u[:, 0:1],
        du_dy=grad_u[:, 1:2],
        dv_dx=grad_v[:, 0:1],
        dv_dy=grad_v[:, 1:2],
    )


def compute_strain(grads: DisplacementGradients) -> Strain:
    """ε_xx = ∂u/∂x, ε_yy = ∂v/∂y, γ_xy = ∂u/∂y + ∂v/∂x."""
    return Strain(
        eps_xx=grads.du_dx,
        eps_yy=grads.dv_dy,
        gamma_xy=grads.du_dy + grads.dv_dx,
    )


def scale_strain(strain: Strain, factor: float | torch.Tensor) -> Strain:
    """Multiply all strain components by a scalar (e.g. U_ref/L_ref)."""
    return Strain(
        eps_xx=factor * strain.eps_xx,
        eps_yy=factor * strain.eps_yy,
        gamma_xy=factor * strain.gamma_xy,
    )
