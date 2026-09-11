"""Analytical and file-based reference solutions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pinn_elasticity.geometry import Rectangle


@dataclass
class ReferenceSolution:
    """Reference fields on an evaluation grid."""

    xy: np.ndarray  # (N, 2)
    u: np.ndarray
    v: np.ndarray
    sigma_xx: np.ndarray | None = None
    sigma_yy: np.ndarray | None = None
    sigma_xy: np.ndarray | None = None
    source: str = "analytical_beam"


def analytical_cantilever_tip_load(
    xy: np.ndarray | torch.Tensor,
    length: float,
    height: float,
    E: float,
    nu: float,
    tip_traction_y: float,
    plane_condition: str = "plane_stress",
) -> ReferenceSolution:
    """
    Timoshenko cantilever beam under end shear (approximate analytical).

    Geometry: x ∈ [0, L], y ∈ [-c, c] in classic formula; here y is shifted
    so the physical domain is y ∈ [0, H] with neutral axis at y = H/2.

    End shear force P = tip_traction_y * H (uniform traction integrated).
    Formulas follow Timoshenko & Goodier (plane stress):

        u = P/(6 EI) (y') [(3L² - 3 L (L-x wait...) classic form with x from fixed end]

    Using:
        I = (2c)^3 / 12 = H^3 / 12
        c = H/2
        y' = y - H/2

        σ_xx = -P (L - x) y' / I
        σ_yy = 0
        σ_xy = -P / (2I) (c² - y'²)

        u = (P y' / (6 E I)) [(6 L - 3 x) x + (2 + ν)(y'² - c²)]
        v = -(P / (6 E I)) [3 ν y'² (L - x) + (4 + 5 ν) c² x + (3 L - x) x² ]

    For plane strain, replace E → E/(1-ν²), ν → ν/(1-ν) in the displacement
    formulas (effective modulus), while stress expressions stay the same for
    this equilibrium field.
    """
    pts = xy.detach().cpu().numpy() if isinstance(xy, torch.Tensor) else np.asarray(xy)
    x = pts[:, 0]
    y = pts[:, 1]
    L = length
    H = height
    c = H / 2.0
    y_p = y - c
    I_zz = H**3 / 12.0
    # Force resultant in +y direction (negative tip_traction_y ⇒ downward load)
    P = tip_traction_y * H

    # Map from classic Timoshenko (P_down > 0) using P_down = -P
    sigma_xx = P * (L - x) * y_p / I_zz
    sigma_yy = np.zeros_like(x)
    sigma_xy = P / (2.0 * I_zz) * (c**2 - y_p**2)

    E_eff, nu_eff = E, nu
    if plane_condition == "plane_strain":
        E_eff = E / (1.0 - nu**2)
        nu_eff = nu / (1.0 - nu)

    u = -(P * y_p / (6.0 * E_eff * I_zz)) * (
        (6.0 * L - 3.0 * x) * x + (2.0 + nu_eff) * (y_p**2 - c**2)
    )
    v = (P / (6.0 * E_eff * I_zz)) * (
        3.0 * nu_eff * y_p**2 * (L - x)
        + (4.0 + 5.0 * nu_eff) * c**2 * x
        + (3.0 * L - x) * x**2
    )

    return ReferenceSolution(
        xy=pts,
        u=u,
        v=v,
        sigma_xx=sigma_xx,
        sigma_yy=sigma_yy,
        sigma_xy=sigma_xy,
        source="analytical_beam",
    )


def load_reference_csv(path: str | Path) -> ReferenceSolution:
    """Load CSV with columns x,y,u,v[,sigma_xx,sigma_yy,sigma_xy]."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    xy = np.column_stack([data["x"], data["y"]])
    ref = ReferenceSolution(xy=xy, u=data["u"], v=data["v"], source="csv")
    names = data.dtype.names or ()
    for key in ("sigma_xx", "sigma_yy", "sigma_xy"):
        if key in names:
            setattr(ref, key, data[key])
    return ref


def build_reference(
    geom: Rectangle,
    E: float,
    nu: float,
    tip_traction_y: float,
    plane_condition: str,
    ref_type: str,
    path: str | None,
    nx: int,
    ny: int,
) -> ReferenceSolution | None:
    """Build reference solution according to configuration."""
    if ref_type == "none":
        return None
    grid = geom.meshgrid(nx, ny).numpy()
    if ref_type == "analytical_beam":
        return analytical_cantilever_tip_load(
            grid,
            length=geom.length,
            height=geom.height,
            E=E,
            nu=nu,
            tip_traction_y=tip_traction_y,
            plane_condition=plane_condition,
        )
    if ref_type == "csv":
        if path is None:
            raise ValueError("reference.path required for csv reference")
        return load_reference_csv(path)
    raise ValueError(f"Unknown reference type: {ref_type}")
