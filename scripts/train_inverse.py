#!/usr/bin/env python3
"""Train an inverse PINN to estimate material parameters from sparse data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.config import load_config, save_config
from pinn_elasticity.evaluation.reference import analytical_cantilever_tip_load
from pinn_elasticity.inverse.parameters import TrainableMaterial
from pinn_elasticity.pipeline import prepare_run, setup_logging
from pinn_elasticity.training.trainer import Trainer
from pinn_elasticity.visualization.plots import plot_loss_history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverse PINN material identification")
    p.add_argument("--config", type=str, default="configs/inverse.yaml")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def synthesize_measurements(config, geom, scales) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse noisy displacement data from analytical beam solution."""
    inv = config.inverse
    rng = np.random.default_rng(config.training.seed + 99)
    x = rng.uniform(0.0, geom.length, size=inv.n_measurements)
    y = rng.uniform(0.0, geom.height, size=inv.n_measurements)
    xy = np.column_stack([x, y])
    ref = analytical_cantilever_tip_load(
        xy,
        length=geom.length,
        height=geom.height,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        tip_traction_y=config.boundary.right_traction_y,
        plane_condition=config.problem.plane_condition,
    )
    u = ref.u + rng.normal(0.0, inv.noise_std * max(np.std(ref.u), 1e-12), size=ref.u.shape)
    v = ref.v + rng.normal(0.0, inv.noise_std * max(np.std(ref.v), 1e-12), size=ref.v.shape)
    uv = np.column_stack([u, v])

    # Save physical measurements
    return (
        torch.as_tensor(xy, dtype=torch.float32),
        torch.as_tensor(uv, dtype=torch.float32),
    )


def load_measurements_csv(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    xy = np.column_stack([data["x"], data["y"]])
    uv = np.column_stack([data["u"], data["v"]])
    return torch.as_tensor(xy, dtype=torch.float32), torch.as_tensor(uv, dtype=torch.float32)


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    if not config.inverse.enabled:
        config.inverse.enabled = True
    config, geom, scales, model, colo, bcs, out = prepare_run(config, run_name=args.run_name)
    save_config(config, out / "config.yaml")

    if config.inverse.data_path:
        xy_phys, uv_phys = load_measurements_csv(Path(config.inverse.data_path))
    else:
        xy_phys, uv_phys = synthesize_measurements(config, geom, scales)

    # Persist measurement CSV
    meas = np.column_stack([xy_phys.numpy(), uv_phys.numpy()])
    np.savetxt(
        out / "measurements.csv",
        meas,
        delimiter=",",
        header="x,y,u,v",
        comments="",
    )

    # Network / losses use non-dimensional coordinates and displacements
    xy_data = scales.to_nondim_coords(xy_phys)
    uv_data = scales.to_nondim_disp(uv_phys)

    E_init = config.inverse.E_init or config.material.youngs_modulus
    nu_init = config.inverse.nu_init or config.material.poisson_ratio
    material = TrainableMaterial(
        E_init=E_init,
        nu_init=nu_init,
        learn_E=config.inverse.learn_E,
        learn_nu=config.inverse.learn_nu,
    )

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

    E_true = config.material.youngs_modulus
    E_est = summary.get("E_est", float("nan"))
    nu_est = summary.get("nu_est", float("nan"))
    rel_E = abs(E_est - E_true) / E_true
    report = {
        "E_true": E_true,
        "E_est": E_est,
        "E_relative_error": rel_E,
        "nu_true": config.material.poisson_ratio,
        "nu_est": nu_est,
        "learn_E": config.inverse.learn_E,
        "learn_nu": config.inverse.learn_nu,
        "noise_std": config.inverse.noise_std,
        "n_measurements": config.inverse.n_measurements,
    }
    with (out / "inverse_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"Inverse complete.\n"
        f"  E_true = {E_true:.6g}\n"
        f"  E_est  = {E_est:.6g}\n"
        f"  rel. error = {rel_E:.4%}\n"
        f"  Outputs: {out}"
    )


if __name__ == "__main__":
    main()
