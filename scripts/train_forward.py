#!/usr/bin/env python3
"""Train a forward PINN for 2D linear elasticity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running without install: python scripts/train_forward.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.config import load_config, save_config
from pinn_elasticity.evaluation.metrics import field_metrics, metrics_table
from pinn_elasticity.evaluation.predict import predict_fields
from pinn_elasticity.evaluation.reference import build_reference
from pinn_elasticity.physics.stress import Stress, von_mises
from pinn_elasticity.pipeline import prepare_run, setup_logging
from pinn_elasticity.training.checkpointing import write_metrics_json
from pinn_elasticity.training.trainer import Trainer
from pinn_elasticity.visualization.plots import plot_all_fields, plot_loss_history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train forward PINN elasticity solver")
    p.add_argument("--config", type=str, default="configs/cantilever.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    config, geom, scales, model, colo, bcs, out = prepare_run(config, run_name=args.run_name)
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
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, indent=2)
    plot_loss_history(summary["history"], out / "figures" / "loss_history.png")

    if args.skip_eval:
        print(f"Training complete. Outputs in {out}")
        return

    # Load best checkpoint
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

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
    xy = torch.as_tensor(ref.xy if ref is not None else geom.meshgrid(51, 21).numpy(), dtype=torch.float32)
    xy_np = xy.numpy() if ref is None else ref.xy

    pred = predict_fields(
        model,
        xy,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        plane_condition=config.problem.plane_condition,
        scales=scales,
        body_force=(config.body_force.fx, config.body_force.fy),
    )

    results = {}
    if ref is not None:
        results["u"] = field_metrics(pred.u, ref.u)
        results["v"] = field_metrics(pred.v, ref.v)
        if ref.sigma_xx is not None:
            results["sigma_xx"] = field_metrics(pred.sigma_xx, ref.sigma_xx)
            results["sigma_yy"] = field_metrics(pred.sigma_yy, ref.sigma_yy)
            results["sigma_xy"] = field_metrics(pred.sigma_xy, ref.sigma_xy)
            vm_ref = von_mises(
                Stress(
                    sigma_xx=torch.as_tensor(ref.sigma_xx).reshape(-1, 1),
                    sigma_yy=torch.as_tensor(ref.sigma_yy).reshape(-1, 1),
                    sigma_xy=torch.as_tensor(ref.sigma_xy).reshape(-1, 1),
                ),
                plane_condition="plane_stress",
            ).numpy().ravel()
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
    plot_all_fields(xy_np, fields, out / "figures")
    print(f"Training + evaluation complete. Outputs in {out}")


if __name__ == "__main__":
    main()
