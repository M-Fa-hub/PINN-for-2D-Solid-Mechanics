"""Field prediction helpers for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pinn_elasticity.physics.constitutive import compute_stress
from pinn_elasticity.physics.equilibrium import compute_equilibrium_residual
from pinn_elasticity.physics.kinematics import (
    compute_displacement_gradients,
    compute_strain,
    scale_strain,
)
from pinn_elasticity.physics.stress import von_mises, von_mises_plane_strain
from pinn_elasticity.scaling import ScaleFactors


@dataclass
class PredictedFields:
    xy: np.ndarray
    u: np.ndarray
    v: np.ndarray
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    gamma_xy: np.ndarray
    sigma_xx: np.ndarray
    sigma_yy: np.ndarray
    sigma_xy: np.ndarray
    von_mises: np.ndarray
    residual_mag: np.ndarray


def _detach_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().reshape(-1)


@torch.enable_grad()
def predict_fields(
    model: nn.Module,
    xy_phys: torch.Tensor,
    E: float | torch.Tensor,
    nu: float | torch.Tensor,
    plane_condition: str,
    scales: ScaleFactors,
    body_force: tuple[float, float] = (0.0, 0.0),
) -> PredictedFields:
    """
    Predict displacement, strain, stress in physical units.

    Model expects non-dimensional coordinates if scaling is enabled.
    """
    model.eval()
    device = next(model.parameters()).device
    xy_phys = xy_phys.to(device)
    xy_in = scales.to_nondim_coords(xy_phys).detach().requires_grad_(True)

    uv_star = model(xy_in)
    grads = compute_displacement_gradients(uv_star, xy_in, create_graph=True)
    strain = scale_strain(compute_strain(grads), scales.strain_scale)
    stress = compute_stress(strain, E, nu, plane_condition=plane_condition)  # type: ignore[arg-type]
    fx, fy = body_force
    L = scales.length_scale
    resid = compute_equilibrium_residual(
        stress, xy_in, body_force=(fx * L, fy * L), create_graph=False
    )

    uv_phys = scales.to_physical_disp(uv_star)
    if plane_condition == "plane_strain":
        vm = von_mises_plane_strain(stress, nu)
    else:
        vm = von_mises(stress, plane_condition="plane_stress")

    return PredictedFields(
        xy=xy_phys.detach().cpu().numpy(),
        u=_detach_np(uv_phys[:, 0]),
        v=_detach_np(uv_phys[:, 1]),
        eps_xx=_detach_np(strain.eps_xx),
        eps_yy=_detach_np(strain.eps_yy),
        gamma_xy=_detach_np(strain.gamma_xy),
        sigma_xx=_detach_np(stress.sigma_xx),
        sigma_yy=_detach_np(stress.sigma_yy),
        sigma_xy=_detach_np(stress.sigma_xy),
        von_mises=_detach_np(vm),
        residual_mag=_detach_np(resid.magnitude()),
    )
