"""Sampling, geometry, network, inverse, config, checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from pinn_elasticity.config import ExperimentConfig, load_config
from pinn_elasticity.geometry import Rectangle
from pinn_elasticity.inverse.parameters import TrainableMaterial
from pinn_elasticity.models.pinn import ElasticityPINN
from pinn_elasticity.sampling import build_collocation, sample_interior
from pinn_elasticity.training.checkpointing import load_checkpoint, save_checkpoint


def test_interior_sampling_bounds() -> None:
    geom = Rectangle(1.0, 0.2)
    pts = sample_interior(geom, 500, method="lhs", seed=1)
    assert pts.shape == (500, 2)
    assert torch.all(pts[:, 0] >= 0) and torch.all(pts[:, 0] <= 1.0)
    assert torch.all(pts[:, 1] >= 0) and torch.all(pts[:, 1] <= 0.2)


def test_boundary_point_generation() -> None:
    geom = Rectangle(2.0, 0.5)
    colo = build_collocation(geom, n_interior=100, n_boundary=80, method="uniform", seed=2)
    assert colo.left.points.shape[0] > 0
    assert torch.allclose(colo.left.points[:, 0], torch.zeros(colo.left.points.shape[0]))
    assert torch.allclose(colo.right.points[:, 0], torch.full((colo.right.points.shape[0],), 2.0))
    assert torch.allclose(colo.bottom.points[:, 1], torch.zeros(colo.bottom.points.shape[0]))
    assert torch.allclose(colo.top.points[:, 1], torch.full((colo.top.points.shape[0],), 0.5))
    total_b = sum(
        s.points.shape[0] for s in (colo.left, colo.right, colo.bottom, colo.top)
    )
    assert total_b == 80


def test_network_output_shape() -> None:
    model = ElasticityPINN(hidden_layers=[32, 32], activation="tanh")
    xy = torch.rand(17, 2)
    out = model(xy)
    assert out.shape == (17, 2)


def test_hard_dirichlet_vanishes_on_left() -> None:
    model = ElasticityPINN(hidden_layers=[16, 16], hard_dirichlet=True)
    xy = torch.zeros(10, 2)
    xy[:, 1] = torch.linspace(0, 1, 10)
    uv = model(xy)
    assert torch.allclose(uv, torch.zeros_like(uv), atol=1e-7)


def test_activations() -> None:
    for act in ("tanh", "sine", "swish"):
        model = ElasticityPINN(hidden_layers=[8], activation=act)  # type: ignore[arg-type]
        assert model(torch.rand(4, 2)).shape == (4, 2)


def test_inverse_parameter_constraints() -> None:
    mat = TrainableMaterial(E_init=1e5, nu_init=0.25, learn_E=True, learn_nu=True)
    for _ in range(20):
        # Push unconstrained params to extremes
        with torch.no_grad():
            mat.theta_E.add_(10.0)
            mat.theta_nu.add_(-5.0)
        E, nu = mat()
        assert E.item() > 0
        assert 0.0 < nu.item() < 0.5


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "geometry": {"length": 1.0, "height": 0.2},
                "material": {"youngs_modulus": -1.0, "poisson_ratio": 0.3},
            }
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "geometry": {"length": 1.0, "height": 0.2},
                "material": {"youngs_modulus": 100.0, "poisson_ratio": 0.6},
            }
        )


def test_load_cantilever_yaml() -> None:
    cfg = load_config("configs/cantilever.yaml")
    assert cfg.problem.plane_condition == "plane_stress"
    assert cfg.geometry.length == 1.0
    assert cfg.scaling.length_ref == cfg.geometry.length


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = ElasticityPINN(hidden_layers=[16, 16])
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, epoch=3, best_loss=0.1, extra={"note": "test"})
    model2 = ElasticityPINN(hidden_layers=[16, 16])
    ckpt = load_checkpoint(path, model2)
    assert ckpt["epoch"] == 3
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
