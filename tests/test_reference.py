"""Reference analytical solution sanity checks."""

from __future__ import annotations

import numpy as np

from pinn_elasticity.evaluation.reference import analytical_cantilever_tip_load


def test_analytical_bc_left_fixed() -> None:
    xy = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.2]])
    ref = analytical_cantilever_tip_load(
        xy, length=1.0, height=0.2, E=210000.0, nu=0.3, tip_traction_y=-1.0
    )
    assert np.allclose(ref.u, 0.0, atol=1e-12)
    # Tip deflection should be downward for negative traction
    tip = analytical_cantilever_tip_load(
        np.array([[1.0, 0.1]]),
        length=1.0,
        height=0.2,
        E=210000.0,
        nu=0.3,
        tip_traction_y=-1.0,
    )
    assert tip.v[0] < 0
