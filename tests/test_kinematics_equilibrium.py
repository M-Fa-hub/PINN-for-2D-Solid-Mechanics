"""Tests for kinematics, AD gradients, and manufactured equilibrium."""

from __future__ import annotations

import torch

from pinn_elasticity.physics.constitutive import compute_stress
from pinn_elasticity.physics.equilibrium import compute_equilibrium_residual
from pinn_elasticity.physics.kinematics import compute_displacement_gradients, compute_strain


def test_strain_from_linear_field() -> None:
    # u = a x + b y, v = c x + d y
    a, b, c, d = 0.01, 0.02, -0.015, 0.03
    n = 64
    xy = torch.rand(n, 2, requires_grad=True)
    u = a * xy[:, 0:1] + b * xy[:, 1:2]
    v = c * xy[:, 0:1] + d * xy[:, 1:2]
    uv = torch.cat([u, v], dim=1)
    grads = compute_displacement_gradients(uv, xy)
    strain = compute_strain(grads)
    assert torch.allclose(strain.eps_xx, torch.full_like(strain.eps_xx, a), atol=1e-6)
    assert torch.allclose(strain.eps_yy, torch.full_like(strain.eps_yy, d), atol=1e-6)
    assert torch.allclose(strain.gamma_xy, torch.full_like(strain.gamma_xy, b + c), atol=1e-6)


def test_manufactured_solution_residual_near_zero() -> None:
    """
    Manufactured quadratic displacement field with matching body force.

    u = A x^2 + B y^2
    v = C x^2 + D y^2

    Strains are linear → stresses linear → divergence is constant.
    Choose body force = -div(σ) so residual ≈ 0.
    """
    A, B, C, D = 1e-3, 2e-3, -1.5e-3, 0.5e-3
    E, nu = 1000.0, 0.3
    n = 200
    torch.manual_seed(0)
    xy = torch.rand(n, 2, requires_grad=True)

    u = A * xy[:, 0:1] ** 2 + B * xy[:, 1:2] ** 2
    v = C * xy[:, 0:1] ** 2 + D * xy[:, 1:2] ** 2
    uv = torch.cat([u, v], dim=1)

    grads = compute_displacement_gradients(uv, xy, create_graph=True)
    strain = compute_strain(grads)
    stress = compute_stress(strain, E, nu, plane_condition="plane_stress")

    # Analytical constant divergence for plane stress Hooke
    # ε_xx = 2 A x, ε_yy = 2 D y, γ_xy = 2 B y + 2 C x
    # σ_xx = factor (2Ax + ν 2Dy), σ_yy = factor (2Dy + ν 2Ax)
    # σ_xy = G (2By + 2Cx)
    # ∂σ_xx/∂x = factor * 2A, ∂σ_xy/∂y = G * 2B
    # ∂σ_xy/∂x = G * 2C, ∂σ_yy/∂y = factor * 2D
    factor = E / (1.0 - nu**2)
    G = E / (2.0 * (1.0 + nu))
    div_x = factor * 2.0 * A + G * 2.0 * B
    div_y = G * 2.0 * C + factor * 2.0 * D
    fx, fy = -div_x, -div_y

    resid = compute_equilibrium_residual(stress, xy, body_force=(fx, fy), create_graph=False)
    assert resid.r_x.shape == (n, 1)
    assert resid.r_y.shape == (n, 1)
    assert torch.mean(resid.r_x**2 + resid.r_y**2).item() < 1e-8


def test_equilibrium_residual_shape() -> None:
    n = 32
    xy = torch.rand(n, 2, requires_grad=True)
    # Quadratic field → non-constant stress with a valid higher-order graph
    uv = torch.cat([xy[:, 0:1] ** 2, xy[:, 1:2] ** 2], dim=1)
    grads = compute_displacement_gradients(uv, xy, create_graph=True)
    strain = compute_strain(grads)
    stress = compute_stress(strain, 100.0, 0.25, plane_condition="plane_stress")
    resid = compute_equilibrium_residual(stress, xy, body_force=(0.0, 0.0))
    assert resid.r_x.shape == (n, 1)
    assert resid.magnitude().shape == (n, 1)
