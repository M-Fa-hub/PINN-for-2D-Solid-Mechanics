"""Tests for constitutive relations and von Mises stress."""

from __future__ import annotations

import torch

from pinn_elasticity.physics.constitutive import compute_stress, lame_from_E_nu
from pinn_elasticity.physics.kinematics import Strain
from pinn_elasticity.physics.stress import Stress, von_mises, von_mises_plane_strain


def test_plane_stress_uniaxial() -> None:
    E, nu = 210000.0, 0.3
    eps = 0.001
    strain = Strain(
        eps_xx=torch.tensor([[eps]]),
        eps_yy=torch.tensor([[-nu * eps]]),
        gamma_xy=torch.tensor([[0.0]]),
    )
    stress = compute_stress(strain, E, nu, plane_condition="plane_stress")
    # Uniaxial: σ_xx ≈ E ε_xx, σ_yy ≈ 0
    assert torch.allclose(stress.sigma_xx, torch.tensor([[E * eps]]), rtol=1e-5)
    assert torch.allclose(stress.sigma_yy, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(stress.sigma_xy, torch.tensor([[0.0]]), atol=1e-12)


def test_plane_strain_bulk() -> None:
    E, nu = 1000.0, 0.25
    lam, mu = lame_from_E_nu(E, nu)
    strain = Strain(
        eps_xx=torch.tensor([[0.01]]),
        eps_yy=torch.tensor([[0.01]]),
        gamma_xy=torch.tensor([[0.0]]),
    )
    stress = compute_stress(strain, E, nu, plane_condition="plane_strain")
    expected = (lam + 2 * mu) * 0.01 + lam * 0.01
    assert torch.allclose(stress.sigma_xx, torch.tensor([[expected]]), rtol=1e-6)
    assert torch.allclose(stress.sigma_yy, torch.tensor([[expected]]), rtol=1e-6)


def test_shear_modulus() -> None:
    E, nu = 200.0, 0.3
    G = E / (2 * (1 + nu))
    strain = Strain(
        eps_xx=torch.tensor([[0.0]]),
        eps_yy=torch.tensor([[0.0]]),
        gamma_xy=torch.tensor([[0.02]]),
    )
    stress = compute_stress(strain, E, nu, plane_condition="plane_stress")
    assert torch.allclose(stress.sigma_xy, torch.tensor([[G * 0.02]]), rtol=1e-6)


def test_von_mises_plane_stress() -> None:
    stress = Stress(
        sigma_xx=torch.tensor([[3.0]]),
        sigma_yy=torch.tensor([[1.0]]),
        sigma_xy=torch.tensor([[2.0]]),
    )
    vm = von_mises(stress, plane_condition="plane_stress")
    expected = (3.0**2 - 3.0 * 1.0 + 1.0**2 + 3.0 * 2.0**2) ** 0.5
    assert torch.allclose(vm, torch.tensor([[expected]]), rtol=1e-6)


def test_von_mises_plane_strain() -> None:
    nu = 0.3
    stress = Stress(
        sigma_xx=torch.tensor([[10.0]]),
        sigma_yy=torch.tensor([[5.0]]),
        sigma_xy=torch.tensor([[0.0]]),
    )
    vm = von_mises_plane_strain(stress, nu)
    szz = nu * (10.0 + 5.0)
    expected = (0.5 * ((10 - 5) ** 2 + (5 - szz) ** 2 + (szz - 10) ** 2)) ** 0.5
    assert torch.allclose(vm, torch.tensor([[expected]]), rtol=1e-5)
