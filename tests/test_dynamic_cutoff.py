import torch

from tugaff.models.energy_head import TaylorEnergyHead
from tugaff.utils.features import smooth_polynomial_cutoff


def _candidate_count(
    max_neighbors: int,
    *,
    dynamic_cutoff: bool,
    multiplier: float = 1.5,
    minimum: int = 16,
) -> int:
    if not dynamic_cutoff:
        return int(max_neighbors)
    return max(int(max_neighbors * multiplier), int(minimum))


def _make_head(**kwargs):
    params = dict(
        d_model=8,
        cutoff=5.0,
        cutoff_width=1.0,
        triplet_cutoff=3.0,
        triplet_gate_width=1.0,
        max_neighbors=2,
        max_neighbors_triplet=1,
        dynamic_cutoff=True,
    )
    params.update(kwargs)
    return TaylorEnergyHead(**params)


def test_tensor_cutoff_matches_scalar_cutoff():
    distances = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32)
    scalar = smooth_polynomial_cutoff(distances, cutoff=2.5, width=1.0)
    tensor = smooth_polynomial_cutoff(
        distances,
        cutoff=torch.full_like(distances, 2.5),
        width=1.0,
    )
    assert torch.allclose(scalar, tensor)


def test_fixed_mode_triplet_support_uses_triplet_cutoff():
    head = TaylorEnergyHead(
        d_model=8,
        cutoff=5.0,
        cutoff_width=1.0,
        triplet_cutoff=2.0,
        triplet_gate_width=0.5,
        dynamic_cutoff=False,
    )
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    triplet_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    distances = torch.tensor([1.0, 1.9, 2.4], dtype=torch.float32)
    _, triplet_support = head.prepare_supports(
        edge_distances=distances,
        edge_index=edge_index,
        triplet_index=triplet_index,
    )
    leg_support = smooth_polynomial_cutoff(
        distances,
        cutoff=2.0,
        width=0.5,
    )
    expected = leg_support[triplet_index[0]] * leg_support[triplet_index[1]]
    assert torch.allclose(triplet_support, expected)


def test_dynamic_cutoff_varies_smoothly_near_neighbor_swap():
    head = _make_head()
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    left = torch.tensor([1.0, 1.49, 1.51], dtype=torch.float32)
    right = torch.tensor([1.0, 1.51, 1.49], dtype=torch.float32)
    support_left, _ = head.prepare_supports(
        edge_distances=left,
        edge_index=edge_index,
        triplet_index=torch.zeros((2, 0), dtype=torch.long),
    )
    support_right, _ = head.prepare_supports(
        edge_distances=right,
        edge_index=edge_index,
        triplet_index=torch.zeros((2, 0), dtype=torch.long),
    )
    assert torch.max(torch.abs(support_left - support_right)) < 0.05


def test_dynamic_support_can_detach_cutoff_solve():
    head = _make_head(max_neighbors=2)
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    distances = torch.tensor([1.0, 1.4, 2.0], dtype=torch.float32, requires_grad=True)
    support, _ = head.prepare_supports(
        edge_distances=distances,
        edge_index=edge_index,
        triplet_index=torch.zeros((2, 0), dtype=torch.long),
        detach_cutoffs=True,
    )
    grad = torch.autograd.grad(support.sum(), distances)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_candidate_count_expands_in_dynamic_mode():
    assert _candidate_count(20, dynamic_cutoff=True, multiplier=1.5, minimum=0) == 30
    assert _candidate_count(9, dynamic_cutoff=True, multiplier=1.5, minimum=16) == 16
    assert _candidate_count(20, dynamic_cutoff=False, multiplier=1.5, minimum=0) == 20
