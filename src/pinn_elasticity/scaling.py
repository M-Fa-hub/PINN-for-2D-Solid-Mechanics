"""Coordinate and field scaling / non-dimensionalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pinn_elasticity.config import ExperimentConfig


@dataclass
class ScaleFactors:
    """
    Scaling convention (starred = network variables):

        x* = x / L_ref
        u* = u / U_ref

    Strain from network gradients must be converted to physical strain:

        ε_phys = (U_ref / L_ref) * ∂u*/∂x*

    Stress uses physical constitutive law with physical E, ν:

        σ_phys = C(E, ν) : ε_phys

    Equilibrium in physical coordinates:

        ∇_x · σ + f = 0,   ∇_x = (1/L_ref) ∇_*

    so the collocation residual is formed as

        r = ∇_* · σ_phys + f * L_ref

    (equivalent up to the constant 1/L_ref).

    Traction BCs use physical σ · n = t.

    Default U_ref ~ |t| L^3 / (E H^2) (cantilever bending scale) so network outputs stay O(1).
    """

    L_ref: float
    U_ref: float
    E_ref: float
    enabled: bool = True

    @property
    def strain_scale(self) -> float:
        """ε_phys = strain_scale * ∂u*/∂x*."""
        if not self.enabled:
            return 1.0
        return self.U_ref / self.L_ref

    @property
    def length_scale(self) -> float:
        return self.L_ref if self.enabled else 1.0

    def to_nondim_coords(self, xy: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return xy
        return xy / self.L_ref

    def to_physical_coords(self, xy_star: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return xy_star
        return xy_star * self.L_ref

    def to_nondim_disp(self, uv: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return uv
        return uv / self.U_ref

    def to_physical_disp(self, uv_star: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return uv_star
        return uv_star * self.U_ref

    def to_physical_stress(self, sigma: torch.Tensor) -> torch.Tensor:
        """Stresses in this formulation are already physical."""
        return sigma

    def to_nondim_stress(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def nondim_E(self, E: float | torch.Tensor) -> float | torch.Tensor:
        """Return physical E (constitutive uses physical modulus)."""
        return E

    def physical_E(self, E: float | torch.Tensor) -> float | torch.Tensor:
        return E


def build_scale_factors(config: ExperimentConfig) -> ScaleFactors:
    """Construct scale factors from experiment config."""
    if not config.scaling.enabled:
        return ScaleFactors(L_ref=1.0, U_ref=1.0, E_ref=1.0, enabled=False)
    assert config.scaling.length_ref is not None
    assert config.scaling.displacement_ref is not None
    assert config.scaling.youngs_ref is not None
    return ScaleFactors(
        L_ref=float(config.scaling.length_ref),
        U_ref=float(config.scaling.displacement_ref),
        E_ref=float(config.scaling.youngs_ref),
        enabled=True,
    )
