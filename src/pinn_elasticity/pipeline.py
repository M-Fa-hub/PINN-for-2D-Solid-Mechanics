"""Shared experiment setup helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

from pinn_elasticity.boundary.conditions import BoundaryConditionSet, build_cantilever_bcs
from pinn_elasticity.config import ExperimentConfig
from pinn_elasticity.geometry import Rectangle
from pinn_elasticity.models.pinn import ElasticityPINN
from pinn_elasticity.sampling import CollocationData, build_collocation
from pinn_elasticity.scaling import ScaleFactors, build_scale_factors


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_geometry(config: ExperimentConfig) -> Rectangle:
    return Rectangle(length=config.geometry.length, height=config.geometry.height)


def make_model(config: ExperimentConfig) -> ElasticityPINN:
    hard = config.boundary.enforcement == "hard"
    return ElasticityPINN(
        hidden_layers=config.network.hidden_layers,
        activation=config.network.activation,
        hard_dirichlet=hard,
        length=config.geometry.length,
    )


def make_collocation(config: ExperimentConfig, geom: Rectangle) -> CollocationData:
    return build_collocation(
        geom,
        n_interior=config.sampling.n_interior,
        n_boundary=config.sampling.n_boundary,
        method=config.sampling.method,
        seed=config.sampling.seed,
    )


def scale_collocation(colo: CollocationData, scales: ScaleFactors) -> CollocationData:
    """Convert collocation coordinates to non-dimensional space for the network."""
    if not scales.enabled:
        return colo
    colo.interior = scales.to_nondim_coords(colo.interior)
    for seg in (colo.left, colo.right, colo.bottom, colo.top):
        seg.points = scales.to_nondim_coords(seg.points)
    return colo


def scale_boundary_tractions(bcs: BoundaryConditionSet, scales: ScaleFactors) -> BoundaryConditionSet:
    """
    Keep Neumann tractions in physical units (σ is physical).

    Dirichlet targets are converted to network displacement units (u*).
    """
    if not scales.enabled:
        return bcs
    for dbc in bcs.dirichlet:
        if not torch.is_tensor(dbc.u):
            dbc.u = float(dbc.u) / scales.U_ref
            dbc.v = float(dbc.v) / scales.U_ref
        else:
            dbc.u = dbc.u / scales.U_ref
            dbc.v = dbc.v / scales.U_ref
    return bcs


def make_bcs(
    config: ExperimentConfig,
    colo: CollocationData,
    geom: Rectangle,
    scales: ScaleFactors,
) -> BoundaryConditionSet:
    return build_cantilever_bcs(
        colo.left,
        colo.right,
        colo.bottom,
        colo.top,
        config.boundary,
        geom=geom,
        scales=scales,
    )


def prepare_run(
    config: ExperimentConfig,
    run_name: str | None = None,
) -> tuple[
    ExperimentConfig,
    Rectangle,
    ScaleFactors,
    ElasticityPINN,
    CollocationData,
    BoundaryConditionSet,
    Path,
]:
    """Assemble model, collocation, BCs, and output directory."""
    set_seed(config.training.seed)
    geom = make_geometry(config)
    scales = build_scale_factors(config)
    model = make_model(config)
    colo = make_collocation(config, geom)
    colo = scale_collocation(colo, scales)
    bcs = make_bcs(config, colo, geom, scales)
    bcs = scale_boundary_tractions(bcs, scales)

    name = run_name or config.logging.run_name or "run"
    out = Path(config.logging.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    return config, geom, scales, model, colo, bcs, out
