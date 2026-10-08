import torch
from unittest.mock import MagicMock
from torch_geometric.data import Batch, Data
from tugaff.models.lit_ff_model import LitTugaForceField

def test_energy_head_weight_decay():
    model = LitTugaForceField(
        num_atom_types=100,
        d_model=32,
        weight_decay=1e-5,
        energy_head_weight_decay=1e-3,
    )
    
    # Mock trainer
    model.trainer = MagicMock()
    model.trainer.estimated_stepping_batches = 1000
    model.trainer.max_epochs = 10
    
    # Call configure_optimizers
    optimizer_dict = model.configure_optimizers()
    optimizer = optimizer_dict["optimizer"]
    
    # Check that optimizer param groups have different weight decay values
    param_groups = optimizer.param_groups
    
    backbone_decay_wd = None
    energy_decay_wd = None
    
    for group in param_groups:
        lr_backbone = model.lr / 10.0
        lr_energy_head = (model.lr * model.hparams.energy_head_lr_factor) / 10.0
        print(f"Group: lr={group['lr']}, weight_decay={group['weight_decay']}, params_len={len(group['params'])}")
        if abs(group['lr'] - lr_backbone) < 1e-9 and group['weight_decay'] > 0:
            backbone_decay_wd = group['weight_decay']
        elif abs(group['lr'] - lr_energy_head) < 1e-9 and group['weight_decay'] > 0:
            energy_decay_wd = group['weight_decay']
    
    print(f"Result: backbone_decay_wd={backbone_decay_wd}, energy_decay_wd={energy_decay_wd}")
    assert backbone_decay_wd == 1e-5
    assert energy_decay_wd == 1e-3


def test_hessian_distance_power_execution():
    # Construct a model with hessian_distance_power set
    model = LitTugaForceField(
        num_atom_types=10,
        d_model=16,
        num_layers=1,
        nhead=2,
        dff_ratio=2,
        num_edge_basis=8,
        num_angle_basis=8,
        hessian_distance_power=1.0,
        hessian_autograd_sample_columns=True,
    )
    
    # Verify configuration
    assert model.hparams.hessian_distance_power == 1.0

    # Construct dummy positions for 3 atoms
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                        [0.0, 2.0, 0.0]], dtype=torch.float32)
    # Fully connected edge index
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    edge_attr = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=-1, keepdim=True)
    
    # Empty triplets
    triplet_index = torch.zeros(2, 0, dtype=torch.long)
    angle_attr = torch.zeros(0, 1)

    # 3 atoms * 3 dof = 9 dof. Hessian is 9x9 = 81 elements
    hessian = torch.randn(81, dtype=torch.float32)

    data = Data(
        x=torch.zeros(3, dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        triplet_index=triplet_index,
        angle_attr=angle_attr,
        has_hessian=torch.tensor([True]),
        hessian=hessian,
    )
    batch = Batch.from_data_list([data])

    # Run _compute_hessian_loss_isolated
    loss = model._compute_hessian_loss_isolated(batch)
    assert loss is not None
    assert loss >= 0.0
