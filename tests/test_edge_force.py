import torch
from torch_geometric.data import Batch, Data


def _tiny_batch(fidelity=None):
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 2.1, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
        dtype=torch.long,
    )
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    edge_attr = torch.linalg.norm(diff, dim=-1, keepdim=True)
    data = Data(
        x=torch.tensor([14, 14, 8], dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        triplet_index=torch.zeros(2, 0, dtype=torch.long),
        angle_attr=torch.zeros(0, 1),
        y=torch.tensor([[0.0]], dtype=torch.float32),
        forces=torch.zeros(3, 3, dtype=torch.float32),
        stress=torch.zeros(1, 3, 3, dtype=torch.float32),
        volume=torch.tensor([20.0], dtype=torch.float32),
    )
    if fidelity is not None:
        data.fidelity = torch.tensor([fidelity], dtype=torch.long)
    return Batch.from_data_list([data])


def _small_model(**kwargs):
    from tugaff.models.ff_model import TugaForceField

    defaults = dict(
        d_model=32,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=12,
        num_angle_basis=8,
        learned_embedding_dim=8,
        node_only_energy=True,
        use_angle_energy=False,
        use_correction_head=False,
        use_edge_force_head=True,
        edge_force_d_hidden=16,
        edge_force_num_mlp_layers=1,
        edge_force_coeff_scale=5.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
    )
    defaults.update(kwargs)
    return TugaForceField(**defaults)


def _set_constant_edge_force(head, raw_bias):
    for p in head.parameters():
        p.data.zero_()
    final_linear = None
    for module in head.mlp.modules():
        if isinstance(module, torch.nn.Linear):
            final_linear = module
    assert final_linear is not None
    final_linear.bias.data.fill_(raw_bias)


def test_edge_force_outputs_forces_and_stress_with_zero_net_force():
    model = _small_model()
    batch = _tiny_batch()

    energy, forces, stress = model(
        batch,
        compute_forces=True,
        compute_stress=True,
        force_mode="edge_force",
        stress_mode="edge_force",
    )

    assert energy.shape == (1, 1)
    assert forces.shape == (batch.num_nodes, 3)
    assert stress.shape == (batch.num_graphs, 3, 3)
    assert torch.isfinite(forces).all()
    assert torch.isfinite(stress).all()
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3), atol=1e-6)


def test_edge_hybrid_outputs_autograd_and_edge_force_branches():
    model = _small_model()
    batch = _tiny_batch()

    energy, forces, stress = model(
        batch,
        compute_forces=True,
        compute_stress=True,
        force_mode="edge_hybrid",
        stress_mode="edge_hybrid",
    )

    assert energy.shape == (1, 1)
    assert set(forces) == {"autograd", "edge_force"}
    assert set(stress) == {"autograd", "edge_force"}
    assert forces["autograd"].shape == (batch.num_nodes, 3)
    assert forces["edge_force"].shape == (batch.num_nodes, 3)
    assert stress["autograd"].shape == (batch.num_graphs, 3, 3)
    assert stress["edge_force"].shape == (batch.num_graphs, 3, 3)
    assert torch.allclose(forces["edge_force"].sum(dim=0), torch.zeros(3), atol=1e-6)


def test_node_only_energy_autograd_forces_still_work():
    model = _small_model()
    batch = _tiny_batch()

    energy, forces = model(batch, compute_forces=True, force_mode="autograd")

    assert energy.shape == (1, 1)
    assert forces.shape == (batch.num_nodes, 3)
    assert torch.isfinite(forces).all()


def test_edge_force_correction_head_is_fidelity_masked():
    model = _small_model(
        use_correction_head=True,
        correction_node_only_energy=True,
        correction_d_hidden=16,
        use_edge_force_correction_head=True,
    )
    _set_constant_edge_force(model.edge_force_head, raw_bias=0.0)
    _set_constant_edge_force(model.edge_force_correction_head, raw_bias=1.0)

    _, f_fid0 = model(
        _tiny_batch(fidelity=0),
        compute_forces=True,
        force_mode="edge_force",
    )
    _, f_fid1 = model(
        _tiny_batch(fidelity=1),
        compute_forces=True,
        force_mode="edge_force",
    )

    assert torch.allclose(f_fid0, torch.zeros_like(f_fid0), atol=1e-6)
    assert f_fid1.abs().max() > 0.0


def test_v31_edge_hybrid_training_step():
    import pytorch_lightning as L
    from torch_geometric.loader import DataLoader
    from tugaff.models.lit_ff_model import LitTugaForceField

    model = LitTugaForceField(
        d_model=32,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=12,
        num_angle_basis=8,
        learned_embedding_dim=8,
        node_only_energy=True,
        use_angle_energy=False,
        use_correction_head=True,
        correction_node_only_energy=True,
        correction_d_hidden=16,
        use_edge_force_head=True,
        use_edge_force_correction_head=True,
        edge_force_d_hidden=16,
        edge_force_num_mlp_layers=1,
        force_mode="edge_hybrid",
        edge_force_weight=0.5,
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
        hessian_interval=0,
        compile_model=False,
    )
    loader = DataLoader([_tiny_batch(fidelity=1).to_data_list()[0]], batch_size=1)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    trainer.fit(model, train_dataloaders=loader)


def test_node_only_param_smoothness_is_noop():
    from tugaff.models.lit_ff_model import LitTugaForceField

    model = LitTugaForceField(
        d_model=32,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=12,
        num_angle_basis=8,
        learned_embedding_dim=8,
        node_only_energy=True,
        use_angle_energy=False,
        use_edge_force_head=True,
        force_mode="edge_hybrid",
        param_smoothness_weight=1.0,
        param_smoothness_mode="structured",
        param_smoothness_max_atoms=8,
        param_smoothness_num_graphs=1,
        param_smoothness_num_perturbations=1,
    )
    batch = _tiny_batch()
    loss, rel, metrics = model._compute_param_smoothness_loss(batch)

    assert loss.item() == 0.0
    assert rel.item() == 0.0
    assert "samples" in metrics
