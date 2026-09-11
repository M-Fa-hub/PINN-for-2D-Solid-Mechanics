"""Boundary condition definitions for 2D elasticity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from pinn_elasticity.config import BoundaryConfig
from pinn_elasticity.geometry import BoundarySegment, Rectangle
from pinn_elasticity.scaling import ScaleFactors


@dataclass
class DirichletBC:
    """Prescribed displacement on a boundary segment."""

    segment: BoundarySegment
    u: float | torch.Tensor = 0.0
    v: float | torch.Tensor = 0.0
    kind: Literal["fixed", "prescribed"] = "fixed"


@dataclass
class NeumannBC:
    """Prescribed traction (t_x, t_y) on a boundary segment."""

    segment: BoundarySegment
    tx: float | torch.Tensor = 0.0
    ty: float | torch.Tensor = 0.0
    kind: Literal["normal", "tangential", "full", "traction_free"] = "full"


@dataclass
class BoundaryConditionSet:
    """Collection of Dirichlet and Neumann conditions for a problem."""

    dirichlet: list[DirichletBC]
    neumann: list[NeumannBC]


def timoshenko_end_shear_traction(
    y_phys: torch.Tensor,
    height: float,
    tip_traction_y: float,
) -> torch.Tensor:
    """
    Parabolic end-shear traction matching the Timoshenko reference.

    With total force P = tip_traction_y * H (force in +y),
        σ_xy(y) = P / (2 I) (c² - y'²),   I = H³/12, c = H/2, y' = y - c

    Returns t_y with shape (N, 1); t_x = 0 on the tip face for this field.
    """
    H = height
    c = H / 2.0
    I_zz = H**3 / 12.0
    P = tip_traction_y * H
    y_p = y_phys.reshape(-1, 1) - c
    return (P / (2.0 * I_zz)) * (c**2 - y_p**2)


def build_cantilever_bcs(
    left: BoundarySegment,
    right: BoundarySegment,
    bottom: BoundarySegment,
    top: BoundarySegment,
    bc_cfg: BoundaryConfig,
    geom: Rectangle | None = None,
    scales: ScaleFactors | None = None,
) -> BoundaryConditionSet:
    """
    Default cantilever:

    - left: fixed (u=v=0)
    - right: uniform or Timoshenko-parabolic tip shear
    - top/bottom: traction-free
    """
    dirichlet = [DirichletBC(segment=left, u=0.0, v=0.0, kind="fixed")]

    if bc_cfg.right_traction_profile == "parabolic":
        if geom is None:
            raise ValueError("geometry required for parabolic tip traction")
        y_coord = right.points[:, 1:2]
        y_phys = y_coord * scales.L_ref if scales is not None and scales.enabled else y_coord
        ty = timoshenko_end_shear_traction(
            y_phys, height=geom.height, tip_traction_y=bc_cfg.right_traction_y
        )
        tx: float | torch.Tensor = torch.zeros_like(ty)
        right_kind: Literal["full"] = "full"
        right_bc = NeumannBC(segment=right, tx=tx, ty=ty, kind=right_kind)
    else:
        right_bc = NeumannBC(
            segment=right,
            tx=bc_cfg.right_traction_x,
            ty=bc_cfg.right_traction_y,
            kind="full",
        )

    neumann = [
        right_bc,
        NeumannBC(segment=top, tx=0.0, ty=0.0, kind="traction_free"),
        NeumannBC(segment=bottom, tx=0.0, ty=0.0, kind="traction_free"),
    ]
    return BoundaryConditionSet(dirichlet=dirichlet, neumann=neumann)
