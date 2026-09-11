#!/usr/bin/env python3
"""Evaluate a trained PINN checkpoint against the reference solution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.config import load_config
from pinn_elasticity.evaluation.metrics import field_metrics, metrics_table
from pinn_elasticity.evaluation.predict import predict_fields
from pinn_elasticity.evaluation.reference import build_reference
from pinn_elasticity.models.pinn import ElasticityPINN
from pinn_elasticity.physics.stress import Stress, von_mises
from pinn_elasticity.pipeline import make_geometry, setup_logging
from pinn_elasticity.scaling import build_scale_factors
from pinn_elasticity.training.checkpointing import write_metrics_json
from pinn_elasticity.visualization.plots import plot_all_fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PINN checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="Override config path")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if args.config:
        config = load_config(args.config)
    else:
        from pinn_elasticity.config import ExperimentConfig

        config = ExperimentConfig.model_validate(ckpt["extra"]["config"])

    geom = make_geometry(config)
    scales = build_scale_factors(config)
    hard = config.boundary.enforcement == "hard"
    model = ElasticityPINN(
        hidden_layers=config.network.hidden_layers,
        activation=config.network.activation,
        hard_dirichlet=hard,
        length=config.geometry.length,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    ref = build_reference(
        geom,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        tip_traction_y=config.boundary.right_traction_y,
        plane_condition=config.problem.plane_condition,
        ref_type=config.reference.type,
        path=config.reference.path,
        nx=config.reference.n_eval_x,
        ny=config.reference.n_eval_y,
    )
    assert ref is not None
    xy = torch.as_tensor(ref.xy, dtype=torch.float32)
    pred = predict_fields(
        model,
        xy,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        plane_condition=config.problem.plane_condition,
        scales=scales,
        body_force=(config.body_force.fx, config.body_force.fy),
    )

    results = {
        "u": field_metrics(pred.u, ref.u),
        "v": field_metrics(pred.v, ref.v),
    }
    if ref.sigma_xx is not None:
        results["sigma_xx"] = field_metrics(pred.sigma_xx, ref.sigma_xx)
        results["sigma_yy"] = field_metrics(pred.sigma_yy, ref.sigma_yy)
        results["sigma_xy"] = field_metrics(pred.sigma_xy, ref.sigma_xy)
        vm_ref = (
            von_mises(
                Stress(
                    torch.as_tensor(ref.sigma_xx).reshape(-1, 1),
                    torch.as_tensor(ref.sigma_yy).reshape(-1, 1),
                    torch.as_tensor(ref.sigma_xy).reshape(-1, 1),
                ),
                plane_condition="plane_stress",
            )
            .numpy()
            .ravel()
        )
        results["von_mises"] = field_metrics(pred.von_mises, vm_ref)

    table = metrics_table(results)
    (out / "metrics_table.md").write_text(table + "\n", encoding="utf-8")
    write_metrics_json(out / "metrics.json", results)
    print(table)

    fields = {
        "u": pred.u,
        "v": pred.v,
        "disp_mag": (pred.u**2 + pred.v**2) ** 0.5,
        "sigma_xx": pred.sigma_xx,
        "sigma_yy": pred.sigma_yy,
        "sigma_xy": pred.sigma_xy,
        "von_mises": pred.von_mises,
        "residual": pred.residual_mag,
    }
    plot_all_fields(ref.xy, fields, out / "figures")
    print(f"Evaluation written to {out}")


if __name__ == "__main__":
    main()
