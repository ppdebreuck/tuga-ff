"""Tests for 3rd-order angular Taylor energy expansion."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

from tugaff.data.dataset import TugaGraphBuilder
from tugaff.models.energy_head import TaylorEnergyHead
from tugaff.models.ff_model import TugaForceField

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_batch(n_graphs=2):
    """Create a small synthetic crystal batch for testing."""
    builder = TugaGraphBuilder(cutoff=5.0, max_neighbors=6, max_neighbors_triplet=6)
    graphs = []
    for i in range(n_graphs):
        lattice = Lattice.cubic(3.0 + 0.1 * i)
        struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g = builder.get_graph(struct, properties={"y": -5.0 + 0.1 * i})
        graphs.append(g)
    return Batch.from_data_list(graphs, follow_batch=["edge_index", "triplet_index"])


# ---------------------------------------------------------------------------
# TaylorEnergyHead with angle energy
# ---------------------------------------------------------------------------


class TestAngleEnergyHead:
    def test_backward_compat(self):
        """Model without angle energy should work exactly as before."""
        head = TaylorEnergyHead(d_model=32, cutoff=8.0)
        assert not head.use_angle_energy
        assert not hasattr(head, "angle_param_mlp")

    def test_angle_mlp_created(self):
        head = TaylorEnergyHead(d_model=32, cutoff=8.0, use_angle_energy=True)
        assert head.use_angle_energy
        assert hasattr(head, "angle_param_mlp")

    def test_forward_with_angles(self):
        head = TaylorEnergyHead(d_model=32, cutoff=8.0, use_angle_energy=True)

        node_feats = torch.randn(3, 32)
        edge_feats = torch.randn(6, 32)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        edge_distances = torch.tensor([2.5, 3.1, 2.5, 2.8, 3.1, 2.8])
        batch_idx = torch.tensor([0, 0, 1])

        triplet_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        angle_values = torch.tensor([0.5, -0.3, 0.8, -0.1]).unsqueeze(-1)
        angle_feats = torch.randn(4, 32)

        energy = head(
            node_feats,
            edge_feats,
            edge_index,
            edge_distances,
            batch_idx,
            2,
            angle_feats=angle_feats,
            angle_values=angle_values,
            triplet_index=triplet_index,
        )
        assert energy.shape == (2, 1)

    def test_forward_without_angles_still_works(self):
        """When use_angle_energy=True but no angle data passed, should not crash."""
        head = TaylorEnergyHead(d_model=32, cutoff=8.0, use_angle_energy=True)

        node_feats = torch.randn(3, 32)
        edge_feats = torch.randn(6, 32)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        edge_distances = torch.tensor([2.5, 3.1, 2.5, 2.8, 3.1, 2.8])
        batch_idx = torch.tensor([0, 0, 1])

        # No angle kwargs passed
        energy = head(
            node_feats,
            edge_feats,
            edge_index,
            edge_distances,
            batch_idx,
            2,
        )
        assert energy.shape == (2, 1)

    def test_gradient_flows_through_angles(self):
        head = TaylorEnergyHead(d_model=32, cutoff=8.0, use_angle_energy=True)

        node_feats = torch.randn(3, 32)
        edge_feats = torch.randn(6, 32)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        edge_distances = torch.tensor([2.5, 3.1, 2.5, 2.8, 3.1, 2.8])
        batch_idx = torch.tensor([0, 0, 1])

        triplet_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        angle_values_leaf = torch.tensor([0.5, -0.3, 0.8, -0.1], requires_grad=True)
        angle_values = angle_values_leaf.unsqueeze(-1)
        angle_feats = torch.randn(4, 32, requires_grad=True)

        energy = head(
            node_feats,
            edge_feats,
            edge_index,
            edge_distances,
            batch_idx,
            2,
            angle_feats=angle_feats,
            angle_values=angle_values,
            triplet_index=triplet_index,
        )
        energy.sum().backward()

        assert angle_feats.grad is not None, "Gradient should flow to angle features"
        assert angle_feats.grad.abs().sum() > 0
        assert angle_values_leaf.grad is not None, (
            "Gradient should flow to angle values"
        )

    def test_angle_cutoff_applied(self):
        """Angles with edges near cutoff should have reduced contribution."""
        head = TaylorEnergyHead(
            d_model=16, cutoff=5.0, cutoff_width=1.0, use_angle_energy=True
        )

        node_feats = torch.randn(2, 16)
        edge_feats = torch.randn(2, 16)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch_idx = torch.tensor([0, 0])

        triplet_index = torch.tensor([[0], [1]])
        angle_values = torch.tensor([0.5]).unsqueeze(-1)
        angle_feats = torch.randn(1, 16)

        # Edge well within cutoff
        e_near = head(
            node_feats,
            edge_feats,
            edge_index,
            torch.tensor([2.0, 2.0]),
            batch_idx,
            1,
            angle_feats=angle_feats,
            angle_values=angle_values,
            triplet_index=triplet_index,
        )

        # Edge at cutoff boundary (angular contribution should be ~0)
        e_far = head(
            node_feats,
            edge_feats,
            edge_index,
            torch.tensor([5.0, 5.0]),
            batch_idx,
            1,
            angle_feats=angle_feats,
            angle_values=angle_values,
            triplet_index=triplet_index,
        )

        # The angular+bond contributions should differ
        assert not torch.allclose(e_near, e_far), (
            "Energy should differ between near and far edges"
        )

    def test_direct_angle_forces_remain_finite_near_linear(self):
        head = TaylorEnergyHead(
            d_model=16,
            cutoff=6.0,
            triplet_cutoff=4.5,
            triplet_gate_width=1.0,
            use_angle_energy=True,
            stop_at_second_order=True,
        )

        angle_feats = torch.randn(1, 16)
        angle_values = torch.tensor([0.9999])
        triplet_index = torch.tensor([[0], [1]])
        edge_index = torch.tensor([[0, 0], [1, 2]])
        edge_distances = torch.tensor([2.0, 2.1])
        diff = torch.tensor([[2.0, 0.0, 0.0], [2.1, 0.01, 0.0]])

        forces, dE_dq, q = head.compute_angle_forces(
            angle_feats=angle_feats,
            angle_values=angle_values,
            triplet_index=triplet_index,
            edge_index=edge_index,
            edge_distances=edge_distances,
            diff=diff,
            num_atoms=3,
            return_derivatives=True,
        )

        assert torch.isfinite(forces).all()
        assert torch.isfinite(dE_dq).all()
        assert torch.isfinite(q).all()

    def test_triplet_d_hidden(self):
        """Test that triplet_d_hidden downscales the triplet parameter MLP properly."""
        head_default = TaylorEnergyHead(
            d_model=32, cutoff=8.0, use_angle_energy=True
        )
        # Default hidden dimension is d_hidden which defaults to d_model (32)
        assert head_default.angle_param_mlp[0].out_features == 32

        head_bottleneck = TaylorEnergyHead(
            d_model=32, cutoff=8.0, use_angle_energy=True, triplet_d_hidden=8
        )
        # Bottleneck hidden dimension should be 8
        assert head_bottleneck.angle_param_mlp[0].out_features == 8



# ---------------------------------------------------------------------------
# Full model with angle energy
# ---------------------------------------------------------------------------


class TestTugaForceFieldAngles:
    @pytest.fixture
    def batch(self):
        return make_batch(n_graphs=2)

    def test_forward_with_angle_energy(self, batch):
        model = TugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            use_angle_energy=True,
        )
        with torch.no_grad():
            e = model(batch)

        assert e.shape == (batch.num_graphs, 1)

    def test_autograd_forces_with_angles(self, batch):
        model = TugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            use_angle_energy=True,
        )
        batch.pos = batch.pos.clone()
        batch.pos[1, 0] = batch.pos[1, 0] + 0.05
        batch.pos.requires_grad_(True)
        e, f = model(batch, compute_forces=True, force_mode="autograd")

        assert f is not None
        assert f.shape == batch.pos.shape
        assert torch.isfinite(f).all(), "Forces should stay finite"

    def test_backward_with_angle_energy(self, batch):
        model = TugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            use_angle_energy=True,
        )
        e = model(batch)
        loss = (e - batch.y.view_as(e)).pow(2).mean()
        loss.backward()

        # Check angle MLP got gradients
        angle_mlp_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.energy_head.angle_param_mlp.parameters()
        )
        assert angle_mlp_has_grad, "Angle param MLP should receive gradients"

    def test_more_params_with_angles(self):
        base = TugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
        )
        angled = TugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            use_angle_energy=True,
        )
        n_base = sum(p.numel() for p in base.parameters())
        n_angled = sum(p.numel() for p in angled.parameters())
        assert n_angled > n_base, "Angle model should have more parameters"


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class TestLitForceFieldAngles:
    def test_training_step(self):
        from tugaff.models.lit_ff_model import LitTugaForceField

        batch = make_batch(n_graphs=4)
        model = LitTugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            lr=1e-3,
            use_angle_energy=True,
        )
        model.trainer = MagicMock()
        model.trainer.max_steps = 100
        model.trainer.optimizers = [MagicMock()]
        model.trainer.optimizers[0].param_groups = [{"lr": 1e-3}]
        with patch.object(type(model), "global_step", new_callable=PropertyMock) as mock_global_step:
            mock_global_step.return_value = 0
            loss = model.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_hparam_saved(self):
        from tugaff.models.lit_ff_model import LitTugaForceField

        model = LitTugaForceField(
            d_model=32,
            num_layers=1,
            nhead=4,
            dff_ratio=2,
            num_edge_basis=20,
            num_angle_basis=10,
            edge_rbf_stop=5.0,
            lr=1e-3,
            use_angle_energy=True,
        )
        assert model.hparams.use_angle_energy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
