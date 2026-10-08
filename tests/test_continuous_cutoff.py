import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax as pyg_softmax

from tugaff.models.energy_head import TaylorEnergyHead
from tugaff.models.ff_model import TugaForceField
from tugaff.models.layers import GraphTransformerLayer, support_weighted_softmax
from tugaff.utils.features import smooth_polynomial_cutoff


def make_two_atom_batch(distance: float, keep_edges: bool = True) -> Batch:
    pos = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float32)
    if keep_edges:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.full((2, 1), distance, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    data = Data(
        x=torch.tensor([1, 1], dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        triplet_index=torch.zeros((2, 0), dtype=torch.long),
        angle_attr=torch.zeros((0, 1), dtype=torch.float32),
        batch=torch.zeros(2, dtype=torch.long),
    )
    return Batch.from_data_list([data])


def test_support_exact_zero_and_derivatives():
    distances = torch.tensor([5.99, 6.0, 6.01])
    support = smooth_polynomial_cutoff(distances, cutoff=6.0, width=1.0)

    assert support[0].item() > 0.0
    assert support[1].item() == 0.0
    assert support[2].item() == 0.0

    head = TaylorEnergyHead(d_model=8, cutoff=6.0, cutoff_width=1.0)
    _, dw, d2w = head._smooth_cutoff_with_second_derivative(distances)
    assert dw[1].item() == 0.0
    assert dw[2].item() == 0.0
    assert d2w[1].item() == 0.0
    assert d2w[2].item() == 0.0


def test_support_weighted_softmax_matches_edge_removal():
    scores = torch.tensor([[0.7], [-0.1], [0.2]], dtype=torch.float32)
    index = torch.tensor([1, 1, 1], dtype=torch.long)
    support = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32)

    weighted = support_weighted_softmax(scores, index, support, num_nodes=3)

    kept = support > 0
    removed = pyg_softmax(
        scores[kept] + support[kept].log().unsqueeze(-1),
        index[kept],
        num_nodes=3,
    )
    expected = torch.zeros_like(weighted)
    expected[kept] = removed

    assert torch.allclose(weighted, expected, atol=1e-7)
    assert weighted[~kept].abs().max().item() == 0.0


def test_graph_attention_zero_support_matches_removed_edge():
    torch.manual_seed(0)
    layer = GraphTransformerLayer(
        d_model=8,
        nhead=2,
        d_edge=8,
        dim_feedforward=16,
        activation="silu",
    ).eval()

    x = torch.randn(3, 8)
    edge_attr = torch.randn(2, 8)
    batch = torch.zeros(3, dtype=torch.long)

    present_edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    removed_edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    out_present = layer(
        x, present_edge_index, edge_attr, batch, support=torch.tensor([1.0, 0.0])
    )
    out_removed = layer(
        x, removed_edge_index, edge_attr[:1], batch, support=torch.tensor([1.0])
    )

    assert torch.allclose(out_present, out_removed, atol=1e-6, rtol=1e-5)


def test_forcefield_dead_edge_matches_removed_edge_beyond_cutoff():
    torch.manual_seed(0)
    model = TugaForceField(
        num_atom_types=10,
        d_model=16,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=16,
        num_angle_basis=8,
        edge_rbf_stop=6.0,
        cutoff_width=1.0,
        learned_embedding_dim=8,
    ).eval()

    retained = make_two_atom_batch(6.01, keep_edges=True)
    removed = make_two_atom_batch(6.01, keep_edges=False)

    with torch.no_grad():
        e_retained = model(retained)
        e_removed = model(removed)

    assert torch.allclose(e_retained, e_removed, atol=1e-6, rtol=1e-5)


def test_forcefield_energy_is_continuous_at_cutoff():
    torch.manual_seed(0)
    model = TugaForceField(
        num_atom_types=10,
        d_model=16,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=16,
        num_angle_basis=8,
        edge_rbf_stop=6.0,
        cutoff_width=1.0,
        learned_embedding_dim=8,
    ).eval()

    near = make_two_atom_batch(5.9999, keep_edges=True)
    at_cutoff = make_two_atom_batch(6.0, keep_edges=True)

    with torch.no_grad():
        e_near = model(near)
        e_cutoff = model(at_cutoff)

    assert torch.allclose(e_near, e_cutoff, atol=5e-3, rtol=1e-4)
