#!/usr/bin/env python3
"""Inverse PINN: estimate E from sparse manufactured displacement data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.boundary.conditions import BoundaryConditionSet, DirichletBC, NeumannBC
from pinn_elasticity.config import load_config, save_config
from pinn_elasticity.evaluation.manufactured import ManufacturedField
from pinn_elasticity.inverse.parameters import TrainableMaterial
from pinn_elasticity.physics.stress import Stress, traction
from pinn_elasticity.pipeline import prepare_run, setup_logging
from pinn_elasticity.training.trainer import Trainer
from pinn_elasticity.visualization.plots import plot_loss_history


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/manufactured_inverse.yaml")
    args = p.parse_args()

    config = load_config(args.config)
    config.inverse.enabled = True
    field = ManufacturedField()
    fx, fy = field.body_force(
        config.material.youngs_modulus,
        config.material.poisson_ratio,
        config.problem.plane_condition,
    )
    config.body_force.fx = fx
    config.body_force.fy = fy
    config.boundary.enforcement = "soft"

    config, geom, scales, model, colo, _, out = prepare_run(
        config, run_name=config.logging.run_name or "manufactured_inverse"
    )

    # Dirichlet + Neumann from manufactured field at true material params
    E_true = config.material.youngs_modulus
    nu_true = config.material.poisson_ratio
    plane = config.problem.plane_condition
    dirichlet = []
    neumann = []
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
        sxx, syy, sxy = field.stress_at(xy_phys, E_true, nu_true, plane)
        tx, ty = traction(Stress(sxx, syy, sxy), seg.normal)
        neumann.append(NeumannBC(segment=seg, tx=tx.detach().clone(), ty=ty.detach().clone()))
    bcs = BoundaryConditionSet(dirichlet=dirichlet, neumann=neumann)

    # Sparse noisy measurements
    rng = np.random.default_rng(config.training.seed + 3)
    n = config.inverse.n_measurements
    xy_phys = torch.as_tensor(
        np.column_stack(
            [
                rng.uniform(0, geom.length, n),
                rng.uniform(0, geom.height, n),
            ]
        ),
        dtype=torch.float32,
    )
    uv_phys = field.displacement(xy_phys).detach()
    noise = config.inverse.noise_std
    uv_phys = uv_phys + noise * torch.randn_like(uv_phys) * uv_phys.std(dim=0, keepdim=True).clamp_min(1e-12)
    np.savetxt(
        out / "measurements.csv",
        np.column_stack([xy_phys.numpy(), uv_phys.numpy()]),
        delimiter=",",
        header="x,y,u,v",
        comments="",
    )
    xy_data = scales.to_nondim_coords(xy_phys)
    uv_data = scales.to_nondim_disp(uv_phys)

    material = TrainableMaterial(
        E_init=config.inverse.E_init or 0.5 * E_true,
        nu_init=config.inverse.nu_init or nu_true,
        learn_E=config.inverse.learn_E,
        learn_nu=config.inverse.learn_nu,
    )
    save_config(config, out / "config.yaml")

    trainer = Trainer(
        model=model,
        config=config,
        collocation=colo,
        bcs=bcs,
        scales=scales,
        geom=geom,
        material=material,
        xy_data=xy_data,
        uv_data=uv_data,
        output_dir=out,
    )
    summary = trainer.fit()
    plot_loss_history(summary["history"], out / "figures" / "loss_history.png")

    E_est = summary["E_est"]
    rel = abs(E_est - E_true) / E_true
    report = {
        "E_true": E_true,
        "E_est": E_est,
        "E_relative_error": rel,
        "nu_true": nu_true,
        "nu_est": summary.get("nu_est"),
    }
    with (out / "inverse_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"E_true={E_true:.6g}  E_est={E_est:.6g}  rel_err={rel:.4%}  -> {out}")


if __name__ == "__main__":
    main()
