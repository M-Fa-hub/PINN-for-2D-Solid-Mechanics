"""Fully connected Physics-Informed Neural Network for 2D elasticity."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn


def _sine(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "sine": _sine,
    "swish": _swish,
}


class ElasticityPINN(nn.Module):
    """
    Map (x, y) → (u, v) with a configurable MLP.

    Soft BC mode: raw network output is the displacement prediction.
    Hard BC mode: apply an output transform that enforces left-edge
    fixed displacement exactly:

        u_hard = x * û(x, y)
        v_hard = x * â(x, y)

    so that u=v=0 at x=0 for a cantilever with left edge fixed.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: Literal["tanh", "sine", "swish"] = "tanh",
        hard_dirichlet: bool = False,
        length: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_name = activation
        self.act = ACTIVATIONS[activation]
        self.hard_dirichlet = hard_dirichlet
        self.length = length

        layers: list[nn.Module] = []
        dims = [2, *hidden_layers, 2]
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward_raw(self, xy: torch.Tensor) -> torch.Tensor:
        """Network output before optional hard BC transform."""
        h = xy
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Predict displacements (u, v).

        xy may be physical or non-dimensional; hard transform uses the
        first coordinate vanishing on the fixed edge (x*=0 when x=0).
        """
        raw = self.forward_raw(xy)
        if not self.hard_dirichlet:
            return raw
        x = xy[:, 0:1]
        # Distance-like factor vanishing on left edge
        return x * raw
