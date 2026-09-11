"""Adam + L-BFGS trainer for elasticity PINNs."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from pinn_elasticity.boundary.conditions import BoundaryConditionSet
from pinn_elasticity.config import ExperimentConfig
from pinn_elasticity.geometry import Rectangle
from pinn_elasticity.inverse.parameters import TrainableMaterial
from pinn_elasticity.sampling import CollocationData, residual_adaptive_sample
from pinn_elasticity.scaling import ScaleFactors
from pinn_elasticity.training.checkpointing import save_checkpoint
from pinn_elasticity.training.losses import LossBreakdown, compute_total_loss

logger = logging.getLogger(__name__)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Trainer:
    """Two-stage PINN trainer with optional inverse parameters and MLflow."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        collocation: CollocationData,
        bcs: BoundaryConditionSet,
        scales: ScaleFactors,
        geom: Rectangle,
        material: TrainableMaterial | None = None,
        xy_data: torch.Tensor | None = None,
        uv_data: torch.Tensor | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.collocation = collocation
        self.bcs = bcs
        self.scales = scales
        self.geom = geom
        self.material = material
        self.xy_data = xy_data
        self.uv_data = uv_data
        self.device = resolve_device(config.training.device)
        self.output_dir = Path(output_dir or config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []
        self._mlflow = None
        if config.logging.backend == "mlflow":
            try:
                import mlflow

                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow requested but not installed; continuing without it.")

        self.model.to(self.device)
        self._move_collocation()
        if self.xy_data is not None:
            self.xy_data = self.xy_data.to(self.device)
            self.uv_data = self.uv_data.to(self.device)  # type: ignore[union-attr]
        if self.material is not None:
            self.material.to(self.device)

    def _move_collocation(self) -> None:
        c = self.collocation
        c.interior = c.interior.to(self.device)
        for seg in (c.left, c.right, c.bottom, c.top):
            seg.points = seg.points.to(self.device)
            seg.normal = seg.normal.to(self.device)
        for nbc in self.bcs.neumann:
            if torch.is_tensor(nbc.tx):
                nbc.tx = nbc.tx.to(self.device)
            if torch.is_tensor(nbc.ty):
                nbc.ty = nbc.ty.to(self.device)
        for dbc in self.bcs.dirichlet:
            if torch.is_tensor(dbc.u):
                dbc.u = dbc.u.to(self.device)
            if torch.is_tensor(dbc.v):
                dbc.v = dbc.v.to(self.device)

    def _E_nu(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if self.material is not None:
            return self.material()
        return self.config.material.youngs_modulus, self.config.material.poisson_ratio

    def _body_force(self) -> tuple[float, float]:
        return self.config.body_force.fx, self.config.body_force.fy

    def _soft_dirichlet(self) -> bool:
        return self.config.boundary.enforcement == "soft"

    def compute_loss(self) -> LossBreakdown:
        E, nu = self._E_nu()
        return compute_total_loss(
            self.model,
            self.collocation.interior,
            self.bcs,
            self.config.loss,
            E,
            nu,
            scales=self.scales,
            plane_condition=self.config.problem.plane_condition,
            body_force=self._body_force(),
            soft_dirichlet=self._soft_dirichlet(),
            xy_data=self.xy_data,
            uv_data=self.uv_data,
        )

    def _log_metrics(self, epoch: int, loss: LossBreakdown, lr: float, elapsed: float) -> None:
        row = {
            "epoch": float(epoch),
            "loss_total": float(loss.total.detach().cpu()),
            "loss_pde": float(loss.pde.detach().cpu()),
            "loss_bc": float(loss.bc.detach().cpu()),
            "loss_data": float(loss.data.detach().cpu()),
            "lr": lr,
            "time_s": elapsed,
        }
        if self.material is not None:
            E_phys, nu = self.material()
            row["E_est"] = float(E_phys.detach().cpu())
            row["nu_est"] = float(nu.detach().cpu())
        self.history.append(row)
        logger.info(
            "epoch=%d total=%.4e pde=%.4e bc=%.4e data=%.4e lr=%.2e t=%.1fs",
            epoch,
            row["loss_total"],
            row["loss_pde"],
            row["loss_bc"],
            row["loss_data"],
            lr,
            elapsed,
        )
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"},
                step=epoch,
            )

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        params = list(self.model.parameters())
        if self.material is not None:
            params.extend(self.material.trainable_parameters())
        return params

    def _maybe_adapt(self, epoch: int) -> None:
        cfg = self.config.adaptive_sampling
        if not cfg.enabled or epoch <= 0 or epoch % cfg.interval != 0:
            return
        from pinn_elasticity.physics.constitutive import compute_stress
        from pinn_elasticity.physics.equilibrium import compute_equilibrium_residual
        from pinn_elasticity.physics.kinematics import (
            compute_displacement_gradients,
            compute_strain,
            scale_strain,
        )

        self.model.eval()
        xy = self.collocation.interior.clone().requires_grad_(True)
        E, nu = self._E_nu()
        uv = self.model(xy)
        grads = compute_displacement_gradients(uv, xy, create_graph=True)
        strain = scale_strain(compute_strain(grads), self.scales.strain_scale)
        stress = compute_stress(
            strain, E, nu, plane_condition=self.config.problem.plane_condition
        )
        fx, fy = self._body_force()
        L = self.scales.length_scale
        resid = compute_equilibrium_residual(
            stress, xy, body_force=(fx * L, fy * L), create_graph=False
        )
        mag = resid.magnitude().reshape(-1)
        # Interior points are stored in the same coordinate system as the network input
        if self.scales.enabled:
            adapt_geom = Rectangle(1.0, self.geom.height / self.scales.L_ref)
            pts_for_adapt = self.collocation.interior.detach()
        else:
            adapt_geom = self.geom
            pts_for_adapt = self.collocation.interior.detach()
        new_pts = residual_adaptive_sample(
            adapt_geom,
            pts_for_adapt,
            mag,
            n_new=cfg.n_new,
            top_fraction=cfg.top_fraction,
            seed=self.config.sampling.seed + epoch,
            dtype=self.collocation.interior.dtype,
        ).to(self.device)
        self.collocation.interior = torch.cat(
            [self.collocation.interior.detach(), new_pts], dim=0
        )
        self.model.train()
        logger.info(
            "Adaptive sampling at epoch %d: added %d points (total interior=%d)",
            epoch,
            cfg.n_new,
            self.collocation.interior.shape[0],
        )

    def train_adam(self) -> float:
        cfg = self.config.training
        params = self._trainable_params()
        opt = torch.optim.Adam(params, lr=cfg.adam_lr)
        if cfg.scheduler == "cosine" and cfg.adam_epochs > 0:
            sched: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.adam_epochs, 1))
            )
        elif cfg.scheduler == "step" and cfg.adam_epochs > 0:
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
            )
        else:
            sched = None

        best = float("inf")
        patience_left = cfg.early_stopping_patience
        t0 = time.perf_counter()
        self.model.train()

        for epoch in range(1, cfg.adam_epochs + 1):
            opt.zero_grad(set_to_none=True)
            loss = self.compute_loss()
            loss.total.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()

            val = float(loss.total.detach().cpu())
            if val < best:
                best = val
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    opt,
                    epoch=epoch,
                    best_loss=best,
                    extra=self._extra_state(),
                )
                if patience_left is not None:
                    patience_left = cfg.early_stopping_patience
            elif patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if epoch % cfg.log_every == 0 or epoch == 1:
                lr = opt.param_groups[0]["lr"]
                self._log_metrics(epoch, loss, lr, time.perf_counter() - t0)

            if epoch % cfg.checkpoint_every == 0:
                save_checkpoint(
                    self.output_dir / f"ckpt_epoch_{epoch}.pt",
                    self.model,
                    opt,
                    epoch=epoch,
                    best_loss=best,
                    extra=self._extra_state(),
                )

            self._maybe_adapt(epoch)

        save_checkpoint(
            self.output_dir / "last.pt",
            self.model,
            opt,
            epoch=cfg.adam_epochs,
            best_loss=best,
            extra=self._extra_state(),
        )
        return best

    def train_lbfgs(self) -> float:
        cfg = self.config.training
        if not cfg.lbfgs_enabled or cfg.lbfgs_max_iter <= 0:
            return float("inf")
        params = self._trainable_params()
        opt = torch.optim.LBFGS(
            params,
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        t0 = time.perf_counter()
        state: dict[str, Any] = {"loss": None}

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = self.compute_loss()
            loss.total.backward()
            state["loss"] = loss
            return loss.total

        self.model.train()
        opt.step(closure)
        loss = state["loss"]
        assert loss is not None
        self._log_metrics(
            epoch=self.config.training.adam_epochs + 1,
            loss=loss,
            lr=cfg.lbfgs_lr,
            elapsed=time.perf_counter() - t0,
        )
        best = float(loss.total.detach().cpu())
        save_checkpoint(
            self.output_dir / "best.pt",
            self.model,
            None,
            epoch=self.config.training.adam_epochs + cfg.lbfgs_max_iter,
            best_loss=best,
            extra=self._extra_state(),
        )
        return best

    def _extra_state(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "config": self.config.model_dump(),
            "scales": {
                "L_ref": self.scales.L_ref,
                "U_ref": self.scales.U_ref,
                "E_ref": self.scales.E_ref,
                "enabled": self.scales.enabled,
            },
        }
        if self.material is not None:
            E, nu = self.material()
            extra["material_est"] = {
                "E": float(E.detach().cpu()),
                "nu": float(nu.detach().cpu()),
            }
        return extra

    def fit(self) -> dict[str, Any]:
        """Run Adam then optional L-BFGS; return summary."""
        if self._mlflow is not None:
            self._mlflow.set_experiment(self.config.logging.experiment_name)
            self._mlflow.start_run(run_name=self.config.logging.run_name)
            self._mlflow.log_params(
                {
                    "plane": self.config.problem.plane_condition,
                    "activation": self.config.network.activation,
                    "n_interior": self.config.sampling.n_interior,
                    "adam_epochs": self.config.training.adam_epochs,
                    "E": self.config.material.youngs_modulus,
                    "nu": self.config.material.poisson_ratio,
                }
            )
        try:
            best_adam = self.train_adam()
            best_lbfgs = self.train_lbfgs()
            summary = {
                "best_adam_loss": best_adam,
                "best_lbfgs_loss": best_lbfgs,
                "output_dir": str(self.output_dir),
                "history": self.history,
            }
            if self.material is not None:
                E, nu = self.material()
                summary["E_est"] = float(E.detach().cpu())
                summary["nu_est"] = float(nu.detach().cpu())
            return summary
        finally:
            if self._mlflow is not None:
                self._mlflow.end_run()
