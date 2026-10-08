"""Lightning module for the simplified Tuga-FF force field."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as L

from .ff_model import TugaForceField


class LitTugaForceField(L.LightningModule):
    """Train `TugaForceField` with energy, autograd forces, and optional stress."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        adam_beta2: float = 0.99,
        loss_type: str = "huber",
        huber_delta: float = 0.1,
        energy_huber_delta: float | None = None,
        force_huber_delta: float | None = None,
        stress_huber_delta: float | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        warmup_pct: float = 0.05,
        onecycle_div_factor: float = 100.0,
        onecycle_final_div_factor: float = 1.0,
        compile_model: bool = False,
        compile_dynamic: bool = True,
        compile_mode: str | None = None,
        deterministic: bool = False,
        head_dtype: str = "fp32",
        float32_matmul_precision: str = "high",
        num_atom_types: int = 100,
        d_model: int = 128,
        edge_rbf_start: float = 0.0,
        edge_rbf_stop: float = 8.0,
        num_edge_basis: int = 80,
        edge_basis_type: str = "gaussian",
        num_angle_basis: int = 40,
        angle_basis_type: str = "cos_uniform",
        num_layers: int = 3,
        nhead: int = 4,
        dff_ratio: float = 4.0,
        activation: str = "silu",
        d_triplet: int | None = None,
        dropout: float = 0.0,
        use_swiglu: bool = False,
        learned_embedding_dim: int = 32,
        cutoff_width: float = 1.0,
        num_energy_mlp_layers: int = 2,
        energy_head_hidden: int | None = None,
        direct_force_head: bool = False,
        direct_force_head_hidden: int | None = None,
        num_direct_force_mlp_layers: int = 2,
        max_neighbors: int | None = 12,
        max_neighbors_triplet: int | None = 12,
        dynamic_cutoff: bool = False,
        element_cutoff_file: str | None = None,
        triplet_cutoff: float = 4.0,
        triplet_gate_width: float = 1.0,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
        edge_update_initial_scale: float = 0.1,
        angle_update_initial_scale: float = 0.0,
        normalize_angle_update: bool = False,
        species_shift_init: str = "zeros",
        support_mode: str = "weighted",
        force_mode: str = "autograd",
        freeze_base_model: bool = False,
        detailed_metrics_interval: int = 100,
        element_weights: torch.Tensor | None = None,
        **deprecated_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["element_weights"])
        if force_mode not in {"autograd", "direct"}:
            raise ValueError("force_mode must be 'autograd' or 'direct'.")
        if force_mode == "direct" and not direct_force_head:
            raise ValueError("force_mode='direct' requires direct_force_head=True.")
        if head_dtype != "fp32":
            raise ValueError("Training currently supports head_dtype='fp32' only.")
        if float32_matmul_precision not in {"high", "highest"}:
            raise ValueError("float32_matmul_precision must be 'high' or 'highest'.")

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.adam_beta2 = float(adam_beta2)
        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0, 1).")
        self.loss_type = loss_type
        self.energy_weight = float(energy_weight)
        self.force_weight = float(force_weight)
        self.stress_weight = float(stress_weight)
        self.huber_delta = float(huber_delta)
        self.energy_huber_delta = float(energy_huber_delta or huber_delta)
        self.force_huber_delta = float(force_huber_delta or huber_delta)
        self.stress_huber_delta = float(stress_huber_delta or huber_delta)
        self.warmup_pct = float(warmup_pct)
        self.onecycle_div_factor = float(onecycle_div_factor)
        self.onecycle_final_div_factor = float(onecycle_final_div_factor)
        self.compile_model = bool(compile_model)
        self.compile_dynamic = bool(compile_dynamic)
        self.compile_mode = compile_mode
        self.deterministic = bool(deterministic)
        self.head_dtype = head_dtype
        self.float32_matmul_precision = float32_matmul_precision
        self.element_weights = element_weights
        self.force_mode = force_mode
        self.freeze_base_model = bool(freeze_base_model)
        self.detailed_metrics_interval = max(1, int(detailed_metrics_interval))

        self.model = TugaForceField(
            num_atom_types=num_atom_types,
            d_model=d_model,
            edge_rbf_start=edge_rbf_start,
            edge_rbf_stop=edge_rbf_stop,
            num_edge_basis=num_edge_basis,
            edge_basis_type=edge_basis_type,
            num_angle_basis=num_angle_basis,
            angle_basis_type=angle_basis_type,
            num_layers=num_layers,
            nhead=nhead,
            dff_ratio=dff_ratio,
            d_triplet=d_triplet,
            activation=activation,
            dropout=dropout,
            use_swiglu=use_swiglu,
            learned_embedding_dim=learned_embedding_dim,
            cutoff_width=cutoff_width,
            num_energy_mlp_layers=num_energy_mlp_layers,
            energy_head_hidden=energy_head_hidden,
            direct_force_head=direct_force_head,
            direct_force_head_hidden=direct_force_head_hidden,
            num_direct_force_mlp_layers=num_direct_force_mlp_layers,
            max_neighbors=max_neighbors,
            max_neighbors_triplet=max_neighbors_triplet,
            dynamic_cutoff=dynamic_cutoff,
            element_cutoff_file=element_cutoff_file,
            triplet_cutoff=triplet_cutoff,
            triplet_gate_width=triplet_gate_width,
            update_hidden_dim=update_hidden_dim,
            nonlinear_updates=nonlinear_updates,
            edge_update_initial_scale=edge_update_initial_scale,
            angle_update_initial_scale=angle_update_initial_scale,
            normalize_angle_update=normalize_angle_update,
            species_shift_init=species_shift_init,
            support_mode=support_mode,
        )
        if self.freeze_base_model:
            self._freeze_base_model()

    def _freeze_base_model(self) -> None:
        """Freeze pretrained weights while keeping the direct readout trainable."""
        if self.model.direct_force_head is None:
            raise ValueError("freeze_base_model requires direct_force_head=True.")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.direct_force_head.parameters():
            parameter.requires_grad = True

    def setup(self, stage: str | None = None) -> None:
        from tugaff.utils.precision import configure_precision

        # Lightning controls backbone autocast. The heads explicitly disable
        # autocast, while PyTorch's native float32 matmul policy controls all
        # eligible FP32 matrix multiplications process-wide.
        configure_precision(
            self.deterministic,
            self.device,
            float32_matmul_precision=self.float32_matmul_precision,
        )
        if self.deterministic:
            self.compile_model = False

        if self.compile_model and hasattr(torch, "compile"):
            if self.force_mode == "direct":
                self.model._compiled_direct = torch.compile(
                    self.model._compute_direct_outputs,
                    dynamic=self.compile_dynamic,
                    mode=self.compile_mode,
                )
            else:
                self.model._compiled_outputs = torch.compile(
                    self.model._compute_outputs,
                    dynamic=self.compile_dynamic,
                    mode=self.compile_mode,
                )

    def _loss(
        self, pred: torch.Tensor, target: torch.Tensor, delta: float
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=pred.device.type,
            enabled=False,
        ):
            pred = (
                pred.float() if pred.dtype in {torch.bfloat16, torch.float16} else pred
            )
            target = target.to(device=pred.device, dtype=pred.dtype).view_as(pred)
            if self.loss_type == "mse":
                return F.mse_loss(pred, target)
            if self.loss_type in {"l1", "mae"}:
                return F.l1_loss(pred, target)
            return F.huber_loss(pred, target, delta=delta)

    @staticmethod
    def _has_attr(batch, name: str) -> bool:
        return hasattr(batch, name) and getattr(batch, name) is not None

    @staticmethod
    def _graph_atom_counts(batch, dtype, device) -> torch.Tensor:
        """Return the number of atoms in each graph in a PyG batch."""
        return (
            torch.bincount(
                batch.batch,
                minlength=int(batch.num_graphs),
            )
            .to(device=device, dtype=dtype)
            .clamp_min(1.0)
        )

    @staticmethod
    def _valid_stress_mask(batch, num_graphs: int, device) -> torch.Tensor:
        """Select graphs carrying real stress labels, not zero placeholders."""
        if not hasattr(batch, "has_stress") or batch.has_stress is None:
            return torch.ones(num_graphs, dtype=torch.bool, device=device)
        mask = torch.as_tensor(batch.has_stress, device=device).bool().view(-1)
        if mask.numel() == 1 and num_graphs != 1:
            mask = mask.expand(num_graphs)
        if mask.numel() != num_graphs:
            raise ValueError(f"Expected {num_graphs} stress flags, got {mask.numel()}.")
        return mask

    def _should_log_detailed_metrics(self, prefix: str) -> bool:
        return (
            prefix != "train" or self.global_step % self.detailed_metrics_interval == 0
        )

    def _log_detailed_metrics(
        self,
        batch,
        prefix: str,
        forces: torch.Tensor | None = None,
        force_target: torch.Tensor | None = None,
    ) -> None:
        """Log physical force diagnostics and graph-size statistics periodically."""
        if not self._should_log_detailed_metrics(prefix):
            return

        num_graphs = max(int(batch.num_graphs), 1)
        num_nodes = max(int(batch.x.size(0)), 1)
        num_edges = int(batch.edge_index.size(1))
        num_triplets = int(batch.triplet_index.size(1))
        stats = {
            "atoms_per_graph": num_nodes / num_graphs,
            "neighbors_per_atom": num_edges / num_nodes,
            "triplets_per_atom": num_triplets / num_nodes,
        }
        for name, value in stats.items():
            value = torch.as_tensor(value, device=batch.x.device, dtype=torch.float32)
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        num_graphs = max(int(batch.num_graphs), 1)
        need_forces = self.force_weight > 0 and self._has_attr(batch, "forces")
        has_stress = self._has_attr(batch, "stress")
        has_valid_stress = (
            has_stress
            and self._valid_stress_mask(batch, num_graphs, batch.x.device).any()
        )
        need_stress = self.stress_weight > 0 and has_valid_stress
        energy, forces, stress = self.model(
            batch,
            compute_forces=need_forces,
            compute_stress=need_stress,
            create_graph_for_forces=self.training,
            force_mode=self.force_mode,
        )

        loss = energy.new_zeros(())
        if self._has_attr(batch, "y") and self.energy_weight > 0:
            energy_target = batch.y.view(-1).to(energy)
            atom_counts = self._graph_atom_counts(
                batch, dtype=energy.dtype, device=energy.device
            )
            energy_loss = self._loss(
                energy / atom_counts,
                energy_target / atom_counts,
                self.energy_huber_delta,
            )
            loss = loss + self.energy_weight * energy_loss
            energy_error = energy - energy_target
            energy_error_per_atom = energy_error / atom_counts
            self.log(
                f"{prefix}/energy_mae_meV_atom",
                energy_error_per_atom.abs().mean() * 1000.0,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=num_graphs,
            )
            if prefix == "val":
                self.log(
                    "val/energy_rmse_meV_atom",
                    energy_error_per_atom.square().mean().sqrt() * 1000.0,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=num_graphs,
                )

        if need_forces and forces is not None:
            force_target = batch.forces.to(device=forces.device, dtype=forces.dtype)
            force_loss = self._loss(forces, force_target, self.force_huber_delta)
            loss = loss + self.force_weight * force_loss
            self.log(
                f"{prefix}/force_mae_meV_A",
                (forces - force_target).abs().mean() * 1000.0,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=max(int(batch.num_nodes), 1),
            )
            if prefix == "val":
                self.log(
                    "val/force_rmse_meV_A",
                    (forces - force_target).square().mean().sqrt() * 1000.0,
                    sync_dist=True,
                    batch_size=max(int(batch.num_nodes), 1),
                )
            self._log_detailed_metrics(batch, prefix)

        if need_stress and stress is not None:
            stress_target = batch.stress.to(device=stress.device, dtype=stress.dtype)
            if stress_target.dim() == 4 and stress_target.shape[1] == 1:
                stress_target = stress_target.squeeze(1)
            stress_mask = self._valid_stress_mask(
                batch, int(batch.num_graphs), stress.device
            )
            if stress_mask.any():
                stress_loss = self._loss(
                    stress[stress_mask],
                    stress_target[stress_mask],
                    self.stress_huber_delta,
                )
                stress_mae_gpa = (
                    stress[stress_mask] - stress_target[stress_mask]
                ).abs().mean() * 160.21766
            else:
                stress_loss = stress.new_zeros(())
                stress_mae_gpa = stress.new_zeros(())
            loss = loss + self.stress_weight * stress_loss
            self.log(
                f"{prefix}/stress_loss",
                stress_loss,
                sync_dist=True,
                batch_size=num_graphs,
            )
            self.log(
                f"{prefix}/stress_mae[GPa]",
                stress_mae_gpa,
                prog_bar=(prefix == "val"),
                sync_dist=True,
                batch_size=num_graphs,
            )

        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        if prefix == "train":
            self.log(
                "train/effective_batch_size",
                float(batch.num_graphs),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=1,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_before_optimizer_step(self, optimizer) -> None:
        """Expose the effective learning rate, including scheduler updates."""
        self.log(
            "train/lr[e-4]",
            float(optimizer.param_groups[0]["lr"]) * 1e4,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

    def configure_optimizers(self):
        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lower_name = name.lower()
            no_decay = (
                lower_name.endswith(".bias")
                or "norm" in lower_name
                or "embedding" in lower_name
                or "species_shift" in lower_name
                or parameter.ndim <= 1
            )
            (no_decay_parameters if no_decay else decay_parameters).append(parameter)

        if not decay_parameters and not no_decay_parameters:
            raise RuntimeError("No trainable parameters are enabled.")
        parameter_groups = []
        if decay_parameters:
            parameter_groups.append(
                {
                    "params": decay_parameters,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
        if no_decay_parameters:
            parameter_groups.append(
                {
                    "params": no_decay_parameters,
                    "lr": self.lr,
                    "weight_decay": 0.0,
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, self.adam_beta2),
            eps=1e-8,
        )
        trainer = getattr(self, "trainer", None)
        max_steps = getattr(trainer, "estimated_stepping_batches", None)
        if not max_steps or max_steps == float("inf"):
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=int(max_steps),
            pct_start=self.warmup_pct,
            div_factor=self.onecycle_div_factor,
            final_div_factor=self.onecycle_final_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(
        self, batch, compute_forces: bool = False, compute_stress: bool = False
    ):
        return self.model(
            batch,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            force_mode=self.force_mode,
        )
