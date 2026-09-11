"""Matplotlib visualization for PINN elasticity results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _contour(
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    triang = tri.Triangulation(xy[:, 0], xy[:, 1])
    tcf = ax.tricontourf(triang, values.ravel(), levels=40, cmap=cmap)
    fig.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_fields(
    xy: np.ndarray,
    fields: dict[str, np.ndarray],
    output_dir: str | Path,
    scale_deform: float | None = None,
) -> list[Path]:
    """
    Generate separate figures for each field plus deformed geometry.

    fields keys expected among:
      u, v, disp_mag, sigma_xx, sigma_yy, sigma_xy, von_mises, residual
    """
    out = _ensure_dir(Path(output_dir))
    saved: list[Path] = []

    mapping = {
        "u": ("Displacement u", "coolwarm"),
        "v": ("Displacement v", "coolwarm"),
        "disp_mag": ("Displacement magnitude", "viridis"),
        "sigma_xx": (r"Stress $\sigma_{xx}$", "RdBu_r"),
        "sigma_yy": (r"Stress $\sigma_{yy}$", "RdBu_r"),
        "sigma_xy": (r"Stress $\sigma_{xy}$", "RdBu_r"),
        "von_mises": ("von Mises stress", "inferno"),
        "residual": ("PDE residual magnitude", "magma"),
    }
    for key, (title, cmap) in mapping.items():
        if key not in fields:
            continue
        path = out / f"{key}.png"
        _contour(xy, fields[key], title, path, cmap=cmap)
        saved.append(path)

    if "u" in fields and "v" in fields:
        path = out / "deformed_geometry.png"
        plot_deformed(xy, fields["u"], fields["v"], path, scale=scale_deform)
        saved.append(path)

    return saved


def plot_deformed(
    xy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    path: Path,
    scale: float | None = None,
) -> None:
    """Scatter undeformed vs deformed mesh (scaled for visibility)."""
    disp = np.sqrt(u.ravel() ** 2 + v.ravel() ** 2)
    if scale is None:
        L = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
        dmax = max(float(disp.max()), 1e-15)
        scale = 0.1 * L / dmax

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.7", label="undeformed", alpha=0.5)
    ax.scatter(
        xy[:, 0] + scale * u.ravel(),
        xy[:, 1] + scale * v.ravel(),
        s=4,
        c=disp,
        cmap="viridis",
        label=f"deformed (×{scale:.2g})",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Deformed geometry")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_history(history: list[dict[str, float]], path: str | Path) -> Path:
    """Plot training loss curves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return path
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("loss_total", "total"),
        ("loss_pde", "PDE"),
        ("loss_bc", "BC"),
        ("loss_data", "data"),
    ]:
        if key in history[0]:
            ax.semilogy(epochs, [h[key] for h in history], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
