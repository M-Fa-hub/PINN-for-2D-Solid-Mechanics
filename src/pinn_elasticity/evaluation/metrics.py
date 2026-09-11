"""Error metrics for PINN vs reference comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def relative_l2(pred: np.ndarray | torch.Tensor, true: np.ndarray | torch.Tensor) -> float:
    """||pred - true||_2 / ||true||_2."""
    p = _to_numpy(pred).ravel()
    t = _to_numpy(true).ravel()
    denom = np.linalg.norm(t)
    if denom < 1e-15:
        return float(np.linalg.norm(p - t))
    return float(np.linalg.norm(p - t) / denom)


def mae(pred: np.ndarray | torch.Tensor, true: np.ndarray | torch.Tensor) -> float:
    p = _to_numpy(pred).ravel()
    t = _to_numpy(true).ravel()
    return float(np.mean(np.abs(p - t)))


def rmse(pred: np.ndarray | torch.Tensor, true: np.ndarray | torch.Tensor) -> float:
    p = _to_numpy(pred).ravel()
    t = _to_numpy(true).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def max_abs_error(pred: np.ndarray | torch.Tensor, true: np.ndarray | torch.Tensor) -> float:
    p = _to_numpy(pred).ravel()
    t = _to_numpy(true).ravel()
    return float(np.max(np.abs(p - t)))


def field_metrics(
    pred: np.ndarray | torch.Tensor,
    true: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    return {
        "relative_l2": relative_l2(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "max_abs": max_abs_error(pred, true),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> str:
    """Format a markdown table of metrics."""
    lines = [
        "| Quantity | Relative L2 Error | RMSE | MAE | Max Abs |",
        "| -------- | ----------------: | ---: | --: | ------: |",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['relative_l2']:.6e} | {m['rmse']:.6e} | "
            f"{m['mae']:.6e} | {m['max_abs']:.6e} |"
        )
    return "\n".join(lines)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_metrics(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {k: dict(v) for k, v in results.items()}
