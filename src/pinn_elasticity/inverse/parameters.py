"""Bounded trainable material parameters for inverse PINNs."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrainableMaterial(nn.Module):
    """
    Safe parameterization of E and ν:

        E  = softplus(θ_E) + eps          → E > 0
        ν  = 0.5 * sigmoid(θ_ν)           → 0 < ν < 0.5

    Only selected parameters are registered as trainable.
    """

    def __init__(
        self,
        E_init: float,
        nu_init: float,
        learn_E: bool = True,
        learn_nu: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if E_init <= 0:
            raise ValueError("E_init must be positive")
        if not (0.0 < nu_init < 0.5):
            raise ValueError("nu_init must satisfy 0 < ν < 0.5")
        self.eps = eps
        self.learn_E = learn_E
        self.learn_nu = learn_nu

        # Stable inverse softplus: log(expm1(y)) overflows for large y; use y + log1p(-e^{-y})
        y = torch.tensor(float(E_init - eps), dtype=torch.float64)
        e_unconstrained = torch.where(
            y > 20.0,
            y,
            torch.log(torch.expm1(y)),
        ).to(dtype=torch.float32)
        # Invert ν = lo + (hi-lo)*sigmoid(θ)
        lo, hi = 1e-4, 0.5 - 1e-4
        ratio = torch.tensor((nu_init - lo) / (hi - lo)).clamp(1e-6, 1.0 - 1e-6)
        nu_unconstrained = torch.log(ratio / (1.0 - ratio))

        if learn_E:
            self.theta_E = nn.Parameter(e_unconstrained)
        else:
            self.register_buffer("theta_E", e_unconstrained)

        if learn_nu:
            self.theta_nu = nn.Parameter(nu_unconstrained)
        else:
            self.register_buffer("theta_nu", nu_unconstrained)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.nn.functional.softplus(self.theta_E) + self.eps
        # Strict open interval (0, 0.5) even at extreme θ
        nu = 1e-4 + (0.5 - 2e-4) * torch.sigmoid(self.theta_nu)
        return E, nu

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.learn_E:
            params.append(self.theta_E)  # type: ignore[arg-type]
        if self.learn_nu:
            params.append(self.theta_nu)  # type: ignore[arg-type]
        return params
