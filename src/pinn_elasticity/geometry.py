"""Rectangular geometry helpers for 2D elasticity domains."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle [0, length] × [0, height]."""

    length: float
    height: float

    def contains(self, xy: torch.Tensor) -> torch.Tensor:
        """Return boolean mask for points inside the closed rectangle."""
        x, y = xy[:, 0], xy[:, 1]
        return (x >= 0) & (x <= self.length) & (y >= 0) & (y <= self.height)

    def corners(self) -> torch.Tensor:
        """Return four corner coordinates as (4, 2)."""
        L, H = self.length, self.height
        return torch.tensor([[0.0, 0.0], [L, 0.0], [L, H], [0.0, H]], dtype=torch.float64)

    def meshgrid(self, nx: int, ny: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Uniform evaluation grid flattened to (nx*ny, 2)."""
        x = torch.linspace(0.0, self.length, nx, dtype=dtype)
        y = torch.linspace(0.0, self.height, ny, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


@dataclass
class BoundarySegment:
    """Named edge of the rectangle with outward unit normal."""

    name: str
    points: torch.Tensor  # (N, 2)
    normal: torch.Tensor  # (2,)
