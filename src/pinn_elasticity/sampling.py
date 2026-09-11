"""Collocation-point sampling for PINN training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc

from pinn_elasticity.geometry import BoundarySegment, Rectangle


@dataclass
class CollocationData:
    """Interior and boundary collocation sets."""

    interior: torch.Tensor
    left: BoundarySegment
    right: BoundarySegment
    bottom: BoundarySegment
    top: BoundarySegment

    def all_boundary_points(self) -> torch.Tensor:
        return torch.cat(
            [self.left.points, self.right.points, self.bottom.points, self.top.points],
            dim=0,
        )


def _to_tensor(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype)


def sample_interior(
    geom: Rectangle,
    n: int,
    method: str = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample n interior points in (0, L) × (0, H)."""
    rng = np.random.default_rng(seed)
    if method == "uniform":
        x = rng.uniform(0.0, geom.length, size=n)
        y = rng.uniform(0.0, geom.height, size=n)
        pts = np.column_stack([x, y])
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        unit = sampler.random(n)
        pts = np.column_stack([unit[:, 0] * geom.length, unit[:, 1] * geom.height])
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return _to_tensor(pts, dtype=dtype)


def sample_boundary(
    geom: Rectangle,
    n_total: int,
    method: str = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> dict[str, BoundarySegment]:
    """
    Sample boundary points distributed across four edges.

    Points are split as evenly as possible among left, right, bottom, top.
    """
    n_per = max(n_total // 4, 1)
    extras = n_total - 4 * n_per
    counts = {
        "left": n_per + (1 if extras > 0 else 0),
        "right": n_per + (1 if extras > 1 else 0),
        "bottom": n_per + (1 if extras > 2 else 0),
        "top": n_per,
    }
    rng = np.random.default_rng(seed + 1)

    def edge_param(n: int, edge_seed: int) -> np.ndarray:
        if method == "uniform":
            return rng.uniform(0.0, 1.0, size=n)
        if method == "lhs":
            sampler = qmc.LatinHypercube(d=1, seed=edge_seed)
            return sampler.random(n).ravel()
        raise ValueError(f"Unknown sampling method: {method}")

    L, H = geom.length, geom.height
    t_left = edge_param(counts["left"], seed + 11)
    t_right = edge_param(counts["right"], seed + 12)
    t_bottom = edge_param(counts["bottom"], seed + 13)
    t_top = edge_param(counts["top"], seed + 14)

    left_pts = np.column_stack([np.zeros(counts["left"]), t_left * H])
    right_pts = np.column_stack([np.full(counts["right"], L), t_right * H])
    bottom_pts = np.column_stack([t_bottom * L, np.zeros(counts["bottom"])])
    top_pts = np.column_stack([t_top * L, np.full(counts["top"], H)])

    return {
        "left": BoundarySegment(
            "left", _to_tensor(left_pts, dtype), torch.tensor([-1.0, 0.0], dtype=dtype)
        ),
        "right": BoundarySegment(
            "right", _to_tensor(right_pts, dtype), torch.tensor([1.0, 0.0], dtype=dtype)
        ),
        "bottom": BoundarySegment(
            "bottom", _to_tensor(bottom_pts, dtype), torch.tensor([0.0, -1.0], dtype=dtype)
        ),
        "top": BoundarySegment(
            "top", _to_tensor(top_pts, dtype), torch.tensor([0.0, 1.0], dtype=dtype)
        ),
    }


def build_collocation(
    geom: Rectangle,
    n_interior: int,
    n_boundary: int,
    method: str = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> CollocationData:
    """Build full collocation dataset."""
    interior = sample_interior(geom, n_interior, method=method, seed=seed, dtype=dtype)
    edges = sample_boundary(geom, n_boundary, method=method, seed=seed, dtype=dtype)
    return CollocationData(
        interior=interior,
        left=edges["left"],
        right=edges["right"],
        bottom=edges["bottom"],
        top=edges["top"],
    )


def residual_adaptive_sample(
    geom: Rectangle,
    existing: torch.Tensor,
    residual_mag: torch.Tensor,
    n_new: int,
    top_fraction: float = 0.2,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Add points near high-residual locations via local random perturbation.

    residual_mag must align with ``existing`` rows (same length).
    """
    if existing.shape[0] != residual_mag.shape[0]:
        raise ValueError("existing and residual_mag length mismatch")
    k = max(1, int(top_fraction * existing.shape[0]))
    idx = torch.topk(residual_mag.detach().reshape(-1), k=k).indices
    hot = existing[idx].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    # Local balls scaled to domain size
    sx = 0.05 * geom.length
    sy = 0.05 * geom.height
    picks = rng.choice(len(hot), size=n_new, replace=True)
    noise = rng.normal(0.0, 1.0, size=(n_new, 2)) * np.array([sx, sy])
    new_pts = hot[picks] + noise
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0.0, geom.length)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0.0, geom.height)
    return _to_tensor(new_pts, dtype=dtype)
