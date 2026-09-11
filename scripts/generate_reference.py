#!/usr/bin/env python3
"""Generate analytical reference CSV for offline comparison / FEM substitute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_elasticity.config import load_config
from pinn_elasticity.evaluation.reference import analytical_cantilever_tip_load
from pinn_elasticity.geometry import Rectangle
from pinn_elasticity.pipeline import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate reference solution CSV")
    p.add_argument("--config", type=str, default="configs/cantilever.yaml")
    p.add_argument("--output", type=str, default="data/reference/cantilever_analytical.csv")
    p.add_argument("--nx", type=int, default=51)
    p.add_argument("--ny", type=int, default=21)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    geom = Rectangle(config.geometry.length, config.geometry.height)
    xy = geom.meshgrid(args.nx, args.ny).numpy()
    ref = analytical_cantilever_tip_load(
        xy,
        length=geom.length,
        height=geom.height,
        E=config.material.youngs_modulus,
        nu=config.material.poisson_ratio,
        tip_traction_y=config.boundary.right_traction_y,
        plane_condition=config.problem.plane_condition,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [ref.xy, ref.u, ref.v, ref.sigma_xx, ref.sigma_yy, ref.sigma_xy]
    )
    np.savetxt(
        out,
        arr,
        delimiter=",",
        header="x,y,u,v,sigma_xx,sigma_yy,sigma_xy",
        comments="",
    )
    print(f"Wrote reference data to {out} ({arr.shape[0]} points)")


if __name__ == "__main__":
    main()
