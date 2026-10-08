"""Core Tuga-FF model: graph transformer backbone plus sum-pooled energy."""

from __future__ import annotations

import csv
import os

import torch
import torch.nn as nn

from ..utils.features import (
    AngleGaussianBasis,
    BesselBasis,
    GaussianSmearing,
    smooth_polynomial_cutoff,
)
from .direct_force_head import DirectForceHead
from .energy_head import SumPoolingEnergyHead
from .layers import InteractionBlock, MLP, prepare_attention_support

try:
    from torch.amp import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast


def _safe_vector_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(v.pow(2).sum(dim=-1) + eps)


def _safe_cosine_similarity(
    v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    n1 = _safe_vector_norm(v1, eps=eps).unsqueeze(-1)
    n2 = _safe_vector_norm(v2, eps=eps).unsqueeze(-1)
    return ((v1 / n1) * (v2 / n2)).sum(dim=-1).clamp(-1.0, 1.0)


class HybridEmbedding(nn.Module):
    """Atom embedding using Pettifor features plus an optional learned part."""

    def __init__(
        self,
        num_atom_types: int = 100,
        d_model: int = 128,
        learned_dim: int = 32,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.learned_embedding = (
            nn.Embedding(num_atom_types + 1, learned_dim) if learned_dim > 0 else None
        )

        properties_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "petiffor_embedding.csv",
        )
        if not os.path.exists(properties_path):
            raise FileNotFoundError(
                f"Pettifor embedding CSV not found: {properties_path}"
            )

        rows: list[list[float]] = []
        with open(properties_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows.append([0.0 if x == "" else float(x) for x in row[1:]])

        feature_dim = len(rows[0]) if rows else 0
        phys = torch.zeros((num_atom_types + 1, feature_dim), dtype=torch.float32)
        for i, vals in enumerate(rows[:num_atom_types], start=1):
            phys[i] = torch.tensor(vals, dtype=torch.float32)
        self.register_buffer("phys_features", phys)

        d_in = feature_dim + (learned_dim if self.learned_embedding is not None else 0)
        self.proj = MLP(
            d_in=d_in,
            d_hidden=d_model,
            d_out=d_model,
            num_layers=2,
            activation=activation,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1).long()
        parts = [self.phys_features[z]]
        if self.learned_embedding is not None:
            parts.append(self.learned_embedding(z))
        return self.norm(self.proj(torch.cat(parts, dim=-1)))


class TugaForceField(nn.Module):
    """Graph transformer force field with a simple extensive energy readout."""

    def __init__(
        self,
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
        d_triplet: int | None = None,
        activation: str = "silu",
        dropout: float = 0.0,
        use_swiglu: bool = False,
        learned_embedding_dim: int = 32,
        cutoff_width: float = 1.0,
        max_neighbors: int | None = 12,
        max_neighbors_triplet: int | None = 12,
        triplet_cutoff: float = 4.0,
        triplet_gate_width: float = 1.0,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
        edge_update_initial_scale: float = 0.1,
        angle_update_initial_scale: float = 0.0,
        normalize_angle_update: bool = False,
        dynamic_cutoff: bool = False,
        element_cutoff_file: str | None = None,
        num_energy_mlp_layers: int = 2,
        energy_head_hidden: int | None = None,
        species_shift_init: str = "zeros",
        direct_force_head: bool = False,
        direct_force_head_hidden: int | None = None,
        num_direct_force_mlp_layers: int = 2,
        support_mode: str = "weighted",
        **deprecated_kwargs,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_triplet = int(d_triplet or d_model)
        self.edge_cutoff = float(edge_rbf_stop)
        self.cutoff_width = float(cutoff_width)
        self.triplet_cutoff = float(triplet_cutoff)
        self.triplet_gate_width = float(triplet_gate_width)
        self.max_neighbors = int(max_neighbors) if max_neighbors is not None else None
        self.max_neighbors_triplet = (
            int(max_neighbors_triplet) if max_neighbors_triplet is not None else None
        )
        self.dynamic_cutoff = bool(dynamic_cutoff)
        self.element_cutoff_file = element_cutoff_file
        self.support_mode = support_mode
        self.nonlinear_updates = bool(nonlinear_updates)

        self.atom_embedding = HybridEmbedding(
            num_atom_types=num_atom_types,
            d_model=d_model,
            learned_dim=learned_embedding_dim,
            activation=activation,
        )

        if edge_basis_type == "bessel":
            self.edge_rbf = BesselBasis(num_basis=num_edge_basis, cutoff=edge_rbf_stop)
        else:
            self.edge_rbf = GaussianSmearing(
                start=edge_rbf_start,
                stop=edge_rbf_stop,
                num_gaussians=num_edge_basis,
            )
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edge_basis, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if angle_basis_type == "theta_uniform":
            self.angle_rbf = AngleGaussianBasis(num_gaussians=num_angle_basis)
        else:
            self.angle_rbf = GaussianSmearing(
                start=-1.0,
                stop=1.0,
                num_gaussians=num_angle_basis,
            )
        self.angle_embedding = nn.Sequential(
            nn.Linear(num_angle_basis, self.d_triplet),
            nn.SiLU(),
            nn.Linear(self.d_triplet, self.d_triplet),
        )

        self.encoder = InteractionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * dff_ratio),
            d_triplet=self.d_triplet,
            num_layers=num_layers,
            activation=activation,
            use_swiglu=use_swiglu,
            dropout=dropout,
            support_mode=support_mode,
            num_edge_basis=num_edge_basis,
            num_angle_basis=num_angle_basis,
            update_hidden_dim=update_hidden_dim,
            nonlinear_updates=self.nonlinear_updates,
            edge_update_initial_scale=edge_update_initial_scale,
            angle_update_initial_scale=angle_update_initial_scale,
            normalize_angle_update=normalize_angle_update,
        )
        self.energy_head = SumPoolingEnergyHead(
            d_model=d_model,
            d_hidden=energy_head_hidden,
            num_atom_types=num_atom_types,
            activation=activation,
            num_mlp_layers=num_energy_mlp_layers,
            species_shift_init=species_shift_init,
        )
        self.direct_force_head = (
            DirectForceHead(
                d_model=d_model,
                d_hidden=direct_force_head_hidden,
                activation=activation,
                num_mlp_layers=num_direct_force_mlp_layers,
            )
            if direct_force_head
            else None
        )
        self._compiled_outputs = None
        self._compiled_direct = None

        if self.dynamic_cutoff:
            from ..utils.elements import load_cutoff_tables

            bond_cutoffs, triplet_cutoffs = load_cutoff_tables(element_cutoff_file)
            self.register_buffer("element_bond_cutoffs", bond_cutoffs.clone())
            self.register_buffer("element_triplet_cutoffs", triplet_cutoffs.clone())

    def update_dropout(self, dropout_rate: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(dropout_rate)

    def _physics_dtype(self) -> torch.dtype:
        configured = getattr(self, "_geometry_dtype", None)
        if configured is not None:
            return configured
        try:
            dtype = next(self.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
        return torch.float64 if dtype == torch.float64 else torch.float32

    def _promote_precision(self, t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None or not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return t
        target = self._physics_dtype()
        if target == torch.float64:
            return t.to(torch.float64)
        if t.dtype in (torch.float16, torch.bfloat16):
            return t.to(torch.float32)
        return t

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _compute_geometry(
        self, batch, pos: torch.Tensor | None, strain: torch.Tensor | None
    ):
        edge_index = batch.edge_index
        edge_dist = batch.edge_attr.squeeze(-1)
        angle_values = batch.angle_attr
        diff = None

        if pos is None:
            return edge_dist, angle_values, diff

        src, dst = edge_index
        pos_active = pos
        lattice = getattr(batch, "lattice_matrix", None)
        if strain is not None:
            pos_active = pos + torch.bmm(
                strain[batch.batch], pos.unsqueeze(-1)
            ).squeeze(-1)
            if lattice is not None:
                lattice = lattice + torch.bmm(lattice, strain.transpose(1, 2))

        diff = pos_active[dst] - pos_active[src]
        if lattice is not None and hasattr(batch, "cell_offsets"):
            edge_batch = batch.batch[src]
            offsets = batch.cell_offsets.to(
                dtype=pos_active.dtype, device=pos_active.device
            )
            shift = torch.bmm(offsets.unsqueeze(1), lattice[edge_batch]).squeeze(1)
            diff = diff + shift

        edge_dist = _safe_vector_norm(diff)
        triplet_index = batch.triplet_index
        if triplet_index.numel() > 0:
            angle_values = _safe_cosine_similarity(
                diff[triplet_index[0]], diff[triplet_index[1]]
            ).unsqueeze(-1)
        else:
            angle_values = batch.angle_attr.to(dtype=edge_dist.dtype)
        return edge_dist, angle_values, diff

    def _supports(
        self,
        edge_dist: torch.Tensor,
        edge_index: torch.Tensor,
        triplet_index: torch.Tensor,
        node_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the support semantics used by the public Taylor model.  The
        # simplified sum-pooling readout still relies on these supports in the
        # GNN backbone, so they are part of checkpoint inference semantics.
        if (
            self.dynamic_cutoff
            and node_z is not None
            and hasattr(self, "element_bond_cutoffs")
            and hasattr(self, "element_triplet_cutoffs")
        ):
            z_src = node_z[edge_index[0]]
            z_dst = node_z[edge_index[1]]
            bond_cutoffs = 0.5 * (
                self.element_bond_cutoffs[z_src] + self.element_bond_cutoffs[z_dst]
            )
            edge_support = smooth_polynomial_cutoff(
                edge_dist,
                cutoff=bond_cutoffs,
                width=self.cutoff_width,
            )
        else:
            edge_support = smooth_polynomial_cutoff(
                edge_dist,
                cutoff=self.edge_cutoff,
                width=self.cutoff_width,
            )

        if triplet_index.numel() == 0:
            triplet_support = edge_dist.new_zeros((0,))
        else:
            if (
                self.dynamic_cutoff
                and node_z is not None
                and hasattr(self, "element_triplet_cutoffs")
            ):
                center_z = node_z[edge_index[0]]
                leg_cutoffs = self.element_triplet_cutoffs[center_z]
            else:
                leg_cutoffs = self.triplet_cutoff

            leg_support = smooth_polynomial_cutoff(
                edge_dist,
                cutoff=leg_cutoffs,
                width=self.triplet_gate_width,
            )
            # A triplet is active only to the extent that both of its legs
            # are active. This is the public calculator's convention.
            triplet_support = (
                leg_support[triplet_index[0]] * leg_support[triplet_index[1]]
            )
        return edge_support, triplet_support

    def _compute_outputs(self, batch, pos=None, strain=None, **kwargs):
        z = batch.x.view(-1).long()
        node_feats = self.atom_embedding(z)

        with autocast(
            device_type=(
                node_feats.device.type if node_feats.device.type != "cpu" else "cpu"
            ),
            enabled=False,
        ):
            edge_dist, angle_values, diff = self._compute_geometry(
                batch,
                self._promote_precision(pos),
                self._promote_precision(strain),
            )
            edge_rbf = self.edge_rbf(self._promote_precision(edge_dist).unsqueeze(-1))
            angle_rbf = self.angle_rbf(self._promote_precision(angle_values))
            edge_support, triplet_support = self._supports(
                self._promote_precision(edge_dist),
                batch.edge_index,
                batch.triplet_index,
                z,
            )

        # Geometry may run in higher precision than the backbone. Cast only at
        # the boundary so distances, angles, basis functions, and smooth cutoff
        # windows retain their requested precision.
        backbone_dtype = node_feats.dtype
        edge_rbf_backbone = edge_rbf.to(dtype=backbone_dtype)
        angle_rbf_backbone = angle_rbf.to(dtype=backbone_dtype)
        edge_support_backbone = edge_support.to(dtype=backbone_dtype)
        triplet_support_backbone = triplet_support.to(dtype=backbone_dtype)
        edge_feats = self.edge_embedding(
            edge_rbf_backbone * edge_support_backbone.unsqueeze(-1)
        )
        angle_feats = self.angle_embedding(angle_rbf_backbone)
        edge_attention_support = prepare_attention_support(edge_support_backbone)
        triplet_attention_support = prepare_attention_support(triplet_support_backbone)

        node_feats, edge_feats = self.encoder(
            node_feats,
            batch.edge_index,
            edge_feats,
            edge_attention_support,
            batch.triplet_index,
            angle_feats,
            triplet_attention_support,
            edge_rbf_backbone,
            angle_rbf_backbone,
        )

        with autocast(
            device_type=(
                node_feats.device.type if node_feats.device.type != "cpu" else "cpu"
            ),
            enabled=False,
        ):
            energy_dtype = self._module_dtype(self.energy_head)
            energy = self.energy_head(
                self._promote_precision(node_feats).to(dtype=energy_dtype),
                z,
                batch.batch,
                num_graphs=batch.num_graphs,
            )
        return energy, node_feats, edge_feats, diff, edge_support

    def _compute_energy(self, batch, pos=None, strain=None, **kwargs):
        """Energy-only entrypoint."""
        energy, _, _, diff, _ = self._compute_outputs(
            batch, pos=pos, strain=strain, **kwargs
        )
        return energy, diff

    def _compute_direct_outputs(self, batch, pos, **kwargs):
        """Compiled direct-inference/training entrypoint including the force head."""
        if self.direct_force_head is None:
            raise RuntimeError("Direct outputs require a direct_force_head.")
        energy, node_feats, edge_feats, diff, edge_support = self._compute_outputs(
            batch, pos=pos, strain=None, **kwargs
        )
        if diff is None:
            raise RuntimeError("Direct forces require positions.")
        forces = self._compute_direct_forces(
            node_feats,
            edge_feats,
            batch.edge_index,
            diff,
            edge_support,
        )
        return energy, forces

    def _compute_direct_forces(
        self,
        node_feats,
        edge_feats,
        edge_index,
        diff,
        edge_support,
    ):
        head_dtype = self._module_dtype(self.direct_force_head)
        with autocast(
            device_type=(
                node_feats.device.type if node_feats.device.type != "cpu" else "cpu"
            ),
            enabled=False,
        ):
            forces = self.direct_force_head(
                node_feats.to(dtype=head_dtype),
                edge_feats.to(dtype=head_dtype),
                edge_index,
                diff.to(dtype=head_dtype),
                edge_support.to(dtype=head_dtype),
            )
        return forces

    def forward(
        self,
        batch,
        compute_forces: bool = False,
        compute_stress: bool = False,
        create_graph_for_forces: bool | None = None,
        force_mode: str = "autograd",
        **kwargs,
    ):
        if force_mode not in {"autograd", "direct"}:
            raise ValueError("force_mode must be 'autograd' or 'direct'.")
        if force_mode == "direct" and self.direct_force_head is None:
            raise RuntimeError(
                "Direct forces requested, but this model has no direct_force_head."
            )
        if force_mode == "direct" and compute_stress:
            raise NotImplementedError("Stress is unavailable when force_mode='direct'.")

        need_autograd = (force_mode == "autograd" and compute_forces) or compute_stress
        need_position = (
            need_autograd or (force_mode == "direct" and compute_forces)
        ) and hasattr(batch, "pos")
        pos = None
        strain = None
        if need_position:
            pos = self._promote_precision(batch.pos)
            if need_autograd:
                pos = pos.detach().clone().requires_grad_(True)
            if compute_stress:
                strain = torch.zeros(
                    (batch.num_graphs, 3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                    requires_grad=True,
                )

        forces = None
        stress = None
        if (
            force_mode == "direct"
            and compute_forces
            and self._compiled_direct is not None
        ):
            energy, forces = self._compiled_direct(batch, pos=pos, **kwargs)
        else:
            outputs_fn = self._compiled_outputs or self._compute_outputs
            energy, node_feats, edge_feats, diff, edge_support = outputs_fn(
                batch, pos=pos, strain=strain, **kwargs
            )
            if force_mode == "direct" and compute_forces:
                if diff is None:
                    raise RuntimeError("Direct forces require batch.pos.")
                forces = self._compute_direct_forces(
                    node_feats,
                    edge_feats,
                    batch.edge_index,
                    diff,
                    edge_support,
                )

        if need_autograd:
            inputs = []
            if compute_forces:
                inputs.append(pos)
            if compute_stress:
                inputs.append(strain)
            create_graph = (
                self.training
                if create_graph_for_forces is None
                else create_graph_for_forces
            )
            grads = torch.autograd.grad(
                energy.sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )
            idx = 0
            if compute_forces:
                forces = -grads[idx].to(dtype=self._module_dtype(self.energy_head))
                idx += 1
            if compute_stress:
                virial = grads[idx].to(dtype=self._module_dtype(self.energy_head))
                if hasattr(batch, "volume"):
                    volume = batch.volume.view(-1, 1, 1).to(
                        dtype=virial.dtype, device=virial.device
                    )
                    stress = virial / volume.clamp_min(1e-12)
                else:
                    stress = virial

        return energy, forces, stress
