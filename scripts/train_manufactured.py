#!/usr/bin/env python3
"""Train a PINN on a manufactured elasticity solution (strong numerical check)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.boundary.conditions import BoundaryConditionSet, DirichletBC, NeumannBC
from pinn_elasticity.config import load_config, save_config
from pinn_elasticity.evaluation.manufactured import ManufacturedField
from pinn_elasticity.evaluation.metrics import field_metrics, metrics_table
from pinn_elasticity.evaluation.predict import predict_fields
from pinn_elasticity.physics.stress import Stress, traction, von_mises
from pinn_elasticity.pipeline import prepare_run, setup_logging
from pinn_elasticity.training.checkpointing import write_metrics_json
from pinn_elasticity.training.trainer import Trainer
from pinn_elasticity.visualization.plots import plot_all_fields, plot_loss_history


def build_manufactured_bcs(
    colo,
    field: ManufacturedField,
    E: float,
    nu: float,
    plane: str,
    scales,
) -> BoundaryConditionSet:
    """Dirichlet on all edges from the manufactured displacement (soft)."""
    dirichlet = []
    for seg in (colo.left, colo.right, colo.bottom, colo.top):
        xy_phys = scales.to_physical_coords(seg.points)
        uv = field.displacement(xy_phys)
        uv_star = scales.to_nondim_disp(uv)
        dirichlet.append(
            DirichletBC(
                segment=seg,
                u=uv_star[:, 0:1].detach().clone(),
                v=uv_star[:, 1:2].detach().clone(),
                kind="prescribed",
            )
        )
    # Also set Neumann from manufactured stress for consistency (optional soft)
    neumann = []
    for seg in (colo.left, colo.right, colo.bottom, colo.top):
        xy_phys = scales.to_physical_coords(seg.points)
        sxx, syy, sxy = field.stress_at(xy_phys, E, nu, plane)
        stress = Stress(sxx, syy, sxy)
        tx, ty = traction(stress, seg.normal)
        neumann.append(NeumannBC(segment=seg, tx=tx.detach().clone(), ty=ty.detach().clone()))
    return BoundaryConditionSet(dirichlet=dirichlet, neumann=neumann)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/manufactured.yaml")
    args = p.parse_args()

    config = load_config(args.config)
    field = ManufacturedField()
    fx, fy = field.body_force(
        config.material.youngs_modulus,
        config.material.poisson_ratio,
        config.problem.plane_condition,
    )
    config.body_force.fx = fx
    config.body_force.fy = fy
    config.boundary.enforcement = "soft"
    config.reference.type = "none"

    config, geom, scales, model, colo, _bcs, out = prepare_run(config, run_name="manufactured")
    bcs = build_manufactured_bcs(
        colo,
        field,
        config.material.youngs_modulus,
        config.material.poisson_ratio,
        config.problem.plane_condition,
        scales,
    )
    save_config(config, out / "config.yaml")

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
    plot_loss_history(summary["history"], out / "figures" / "loss_history.png")

    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    xy = geom.meshgrid(31, 21)
    ref_uv = field.displacement(xy)
    sxx, syy, sxy = field.stress_at(
        xy, config.material.youngs_modulus, config.material.poisson_ratio, config.problem.plane_condition
    )
    pred = predict_fields(
        model,
        xy,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        plane_condition=config.problem.plane_condition,
        scales=scales,
        body_force=(fx, fy),
    )
    vm_ref = von_mises(Stress(sxx, syy, sxy), plane_condition="plane_stress").detach().numpy().ravel()
    results = {
        "u": field_metrics(pred.u, ref_uv[:, 0].numpy()),
        "v": field_metrics(pred.v, ref_uv[:, 1].numpy()),
        "sigma_xx": field_metrics(pred.sigma_xx, sxx.detach().numpy()),
        "sigma_yy": field_metrics(pred.sigma_yy, syy.detach().numpy()),
        "sigma_xy": field_metrics(pred.sigma_xy, sxy.detach().numpy()),
        "von_mises": field_metrics(pred.von_mises, vm_ref),
    }
    table = metrics_table(results)
    (out / "metrics_table.md").write_text(table + "\n", encoding="utf-8")
    write_metrics_json(out / "metrics.json", results)
    print(table)
    plot_all_fields(
        xy.numpy(),
        {
            "u": pred.u,
            "v": pred.v,
            "disp_mag": (pred.u**2 + pred.v**2) ** 0.5,
            "sigma_xx": pred.sigma_xx,
            "sigma_yy": pred.sigma_yy,
            "sigma_xy": pred.sigma_xy,
            "von_mises": pred.von_mises,
            "residual": pred.residual_mag,
        },
        out / "figures",
    )
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, indent=2)
    print(f"Manufactured run complete: {out}")


if __name__ == "__main__":
    main()
