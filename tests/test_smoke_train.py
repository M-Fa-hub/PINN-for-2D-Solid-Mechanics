"""Smoke test: short forward training run."""

from __future__ import annotations

from pathlib import Path

import torch

from pinn_elasticity.config import load_config
from pinn_elasticity.evaluation.predict import predict_fields
from pinn_elasticity.pipeline import prepare_run
from pinn_elasticity.training.trainer import Trainer


def test_smoke_forward_training(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.logging.output_dir = str(tmp_path)
    config, geom, scales, model, colo, bcs, out = prepare_run(config, run_name="pytest_smoke")
    trainer = Trainer(
        model=model,
        config=config,
        collocation=colo,
        bcs=bcs,
        scales=scales,
        geom=geom,
        output_dir=out,
    )
    summary = trainer.fit()
    assert (out / "best.pt").is_file()
    assert summary["best_adam_loss"] < 1e6

    xy = geom.meshgrid(11, 5)
    pred = predict_fields(
        model,
        xy,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        plane_condition=config.problem.plane_condition,
        scales=scales,
    )
    assert pred.u.shape[0] == 11 * 5
    assert torch.isfinite(torch.as_tensor(pred.von_mises)).all()
