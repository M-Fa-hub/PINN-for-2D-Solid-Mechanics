"""Configuration models and YAML loading for PINN elasticity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class GeometryConfig(BaseModel):
    """Rectangular domain geometry [0, L] x [0, H]."""

    length: float = Field(gt=0, description="Domain length L in x-direction")
    height: float = Field(gt=0, description="Domain height H in y-direction")


class MaterialConfig(BaseModel):
    """
    Linear elastic isotropic material parameters.

    Use a consistent unit system throughout (e.g. N, mm, MPa or N, m, Pa).
    Default values use a scaled system suitable for PINN training:
    E = 210000 (e.g. MPa if lengths are in mm), ν = 0.3.
    """

    youngs_modulus: float = Field(gt=0, description="Young's modulus E > 0")
    poisson_ratio: float = Field(gt=0, lt=0.5, description="Poisson ratio 0 < ν < 0.5")


class NetworkConfig(BaseModel):
    """Fully connected PINN architecture."""

    hidden_layers: list[int] = Field(default_factory=lambda: [64, 64, 64, 64])
    activation: Literal["tanh", "sine", "swish"] = "tanh"

    @field_validator("hidden_layers")
    @classmethod
    def _nonempty_positive(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("hidden_layers must be non-empty")
        if any(n <= 0 for n in v):
            raise ValueError("hidden_layers widths must be positive")
        return v


class SamplingConfig(BaseModel):
    """Collocation and boundary sampling."""

    n_interior: int = Field(default=10000, ge=1)
    n_boundary: int = Field(default=2000, ge=4)
    method: Literal["uniform", "lhs"] = "lhs"
    seed: int = 42


class AdaptiveSamplingConfig(BaseModel):
    """Optional residual-based adaptive refinement."""

    enabled: bool = False
    interval: int = Field(default=2000, ge=1)
    n_new: int = Field(default=1000, ge=1)
    top_fraction: float = Field(default=0.2, gt=0, le=1)


class ScalingConfig(BaseModel):
    """
    Non-dimensionalization / normalization scales.

    Physical quantities map as:
        x* = x / L_ref,   y* = y / L_ref
        u* = u / U_ref,   v* = v / U_ref
        σ* = σ / E_ref
        f* = f * L_ref / E_ref

    Network operates in starred (non-dimensional) variables.
    Set L_ref to geometry length and E_ref to Young's modulus by default.
    """

    enabled: bool = True
    length_ref: float | None = Field(
        default=None, description="Reference length L_ref; defaults to geometry.length"
    )
    displacement_ref: float | None = Field(
        default=None,
        description="Reference displacement U_ref; defaults to L_ref * traction_scale / E",
    )
    youngs_ref: float | None = Field(
        default=None, description="Reference modulus E_ref; defaults to material.E"
    )


class LossConfig(BaseModel):
    """Weighted multi-objective loss coefficients."""

    lambda_pde: float = Field(default=1.0, ge=0)
    lambda_bc: float = Field(default=100.0, ge=0)
    lambda_data: float = Field(default=1.0, ge=0)
    lambda_dirichlet: float = Field(default=1.0, ge=0)
    lambda_neumann: float = Field(default=1.0, ge=0)


class TrainingConfig(BaseModel):
    """Two-stage Adam → L-BFGS training settings."""

    adam_epochs: int = Field(default=10000, ge=0)
    adam_lr: float = Field(default=1e-3, gt=0)
    lbfgs_enabled: bool = True
    lbfgs_max_iter: int = Field(default=500, ge=0)
    lbfgs_lr: float = Field(default=1.0, gt=0)
    scheduler: Literal["none", "cosine", "step"] = "cosine"
    scheduler_step_size: int = 2000
    scheduler_gamma: float = 0.5
    early_stopping_patience: int | None = Field(default=None, ge=1)
    grad_clip: float | None = Field(default=1.0, gt=0)
    log_every: int = Field(default=100, ge=1)
    eval_every: int = Field(default=500, ge=1)
    checkpoint_every: int = Field(default=1000, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42


class BoundaryConfig(BaseModel):
    """Cantilever BC parameters."""

    left: Literal["fixed"] = "fixed"
    right_traction_x: float = 0.0
    right_traction_y: float = -1.0
    # parabolic matches the Timoshenko analytical end-shear; uniform is a different BVP
    right_traction_profile: Literal["uniform", "parabolic"] = "parabolic"
    top: Literal["traction_free"] = "traction_free"
    bottom: Literal["traction_free"] = "traction_free"
    enforcement: Literal["soft", "hard"] = "soft"


class BodyForceConfig(BaseModel):
    """Constant body force density (f_x, f_y)."""

    fx: float = 0.0
    fy: float = 0.0


class ReferenceConfig(BaseModel):
    """Reference / analytical comparison settings."""

    type: Literal["analytical_beam", "csv", "none"] = "analytical_beam"
    path: str | None = None
    n_eval_x: int = Field(default=51, ge=5)
    n_eval_y: int = Field(default=21, ge=5)


class InverseConfig(BaseModel):
    """Inverse material-parameter estimation."""

    enabled: bool = False
    learn_E: bool = True
    learn_nu: bool = False
    E_init: float | None = None
    nu_init: float | None = None
    data_path: str | None = None
    noise_std: float = Field(default=0.0, ge=0)
    n_measurements: int = Field(default=50, ge=1)


class LoggingConfig(BaseModel):
    """Experiment tracking."""

    backend: Literal["none", "mlflow", "tensorboard"] = "none"
    experiment_name: str = "pinn-elasticity"
    run_name: str | None = None
    output_dir: str = "outputs"


class ProblemConfig(BaseModel):
    """Top-level problem definition."""

    type: Literal["cantilever"] = "cantilever"
    plane_condition: Literal["plane_stress", "plane_strain"] = "plane_stress"


class ExperimentConfig(BaseModel):
    """Full experiment configuration loaded from YAML."""

    problem: ProblemConfig = Field(default_factory=ProblemConfig)
    geometry: GeometryConfig
    material: MaterialConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    adaptive_sampling: AdaptiveSamplingConfig = Field(default_factory=AdaptiveSamplingConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    body_force: BodyForceConfig = Field(default_factory=BodyForceConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    inverse: InverseConfig = Field(default_factory=InverseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _resolve_scaling_defaults(self) -> ExperimentConfig:
        if self.scaling.enabled:
            if self.scaling.length_ref is None:
                self.scaling.length_ref = self.geometry.length
            if self.scaling.youngs_ref is None:
                self.scaling.youngs_ref = self.material.youngs_modulus
            if self.scaling.displacement_ref is None:
                L = self.scaling.length_ref
                H = self.geometry.height
                E = self.scaling.youngs_ref
                t_mag = max(
                    abs(self.boundary.right_traction_x),
                    abs(self.boundary.right_traction_y),
                    1e-12,
                )
                # Bending-dominated cantilever scale ~ t L^3 / (E H^2)
                # (much larger than simple axial scale t L / E when L >> H)
                self.scaling.displacement_ref = t_mag * (L**3) / (E * H**2)
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write configuration to YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False, default_flow_style=False)
