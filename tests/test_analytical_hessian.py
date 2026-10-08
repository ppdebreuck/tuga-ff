"""Sanity tests for compute_analytical_hessian.

Checks:
  1. Symmetry:               H == H.T
  2. Translational inv.:     each block-row sums to zero (acoustic sum rule)
  3. Numerical consistency:  H_analytical ≈ H_numerical (finite-diff on direct forces)

Run with:
    python tests/test_analytical_hessian.py
or:
    pytest tests/test_analytical_hessian.py -v
"""

import math
import sys
import torch
import pytest
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# Helper: build a minimal TugaForceField with small random weights
# ---------------------------------------------------------------------------

def make_model(
    use_morse=True,
    use_angles=False,
    use_stretch_bend=False,
    analytical_hessian_detach_angle_features=True,
):
    from tugaff.models.ff_model import TugaForceField
    return TugaForceField(
        num_atom_types=10,
        d_model=32,
        num_layers=1,
        nhead=4,
        dff_ratio=2,
        num_edge_basis=16,
        edge_rbf_start=0.0,
        edge_rbf_stop=6.0,
        num_angle_basis=16,
        edge_basis_type="bessel",
        learned_embedding_dim=16,
        num_energy_mlp_layers=2,
        cutoff_width=1.0,
        use_morse_envelope=use_morse,
        use_angle_energy=use_angles,
        use_stretch_bend=use_stretch_bend,
        stable_hessian=True,
        analytical_hessian_detach_angle_features=analytical_hessian_detach_angle_features,
    ).eval()


# ---------------------------------------------------------------------------
# Helper: build a minimal PyG batch (N atoms, fully connected within cutoff)
# ---------------------------------------------------------------------------

def make_batch(pos, atom_types=None, cutoff=5.0):
    """
    pos: (N, 3) float32 positions
    Returns a PyG Batch containing edges for all pairs within cutoff.
    """
    N = pos.shape[0]
    if atom_types is None:
        atom_types = torch.zeros(N, dtype=torch.long)

    src_list, dst_list = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = (pos[j] - pos[i]).norm()
            if d < cutoff:
                src_list.append(i)
                dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = (pos[edge_index[1]] - pos[edge_index[0]]).norm(dim=-1, keepdim=True)

    # Triplets: all pairs of edges sharing the same src
    from collections import defaultdict
    src_to_edges = defaultdict(list)
    for e, s in enumerate(src_list):
        src_to_edges[s].append(e)

    triplet_left, triplet_right = [], []
    for edges in src_to_edges.values():
        for a in range(len(edges)):
            for b in range(len(edges)):
                if a != b:
                    triplet_left.append(edges[a])
                    triplet_right.append(edges[b])

    if triplet_left:
        triplet_index = torch.tensor([triplet_left, triplet_right], dtype=torch.long)
        angle_vecs_j = (pos[edge_index[1, triplet_index[0]]] - pos[edge_index[0, triplet_index[0]]])
        angle_vecs_k = (pos[edge_index[1, triplet_index[1]]] - pos[edge_index[0, triplet_index[1]]])
        angle_attr = torch.cosine_similarity(angle_vecs_j, angle_vecs_k, dim=-1, eps=1e-8).unsqueeze(-1)
    else:
        triplet_index = torch.zeros(2, 0, dtype=torch.long)
        angle_attr = torch.zeros(0, 1)

    data = Data(
        x=atom_types,
        pos=pos.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        triplet_index=triplet_index,
        angle_attr=angle_attr,
        batch=torch.zeros(N, dtype=torch.long),
    )
    return Batch.from_data_list([data])


# ---------------------------------------------------------------------------
# Numerical Hessian via central differences on frozen-feature direct forces
# ---------------------------------------------------------------------------
# compute_analytical_hessian freezes GNN features at the equilibrium geometry
# and differentiates only the analytical bond/angle energy formulas.
# The correct numerical reference uses the same frozen-features approximation:
# perturb positions, recompute geometry, but keep GNN outputs fixed.
def detached_feature_cache(model, batch):
    cached = model._gnn_features(batch)
    z = batch.x.view(-1) if batch.x.dim() > 1 else batch.x
    return {
        "node_feats": cached["node_feats"].detach(),
        "edge_feats": cached["edge_feats"].detach(),
        "angle_feats": cached["angle_feats"].detach(),
        "edge_rbf": cached["edge_rbf"].detach(),
        "edge_index": cached["edge_index"],
        "triplet_index": cached["triplet_index"],
        "batch_idx": cached["batch_idx"],
        "num_graphs": cached["num_graphs"],
        "node_z": z,
    }

@torch.no_grad()
def numerical_hessian_frozen(model, batch, eps=1e-3):
    """Numerical H[αi, βj] = -dF_i^α/dr_j^β using frozen GNN features.

    Equivalent to what compute_analytical_hessian approximates:
    d(analytical_forces(frozen_GNN_features, r)) / dr.
    """
    cached = detached_feature_cache(model, batch)

    pos0 = batch.pos.clone().float()
    N = pos0.shape[0]
    edge_index = cached["edge_index"]
    triplet_indices = cached["triplet_index"]
    batch_idx = cached["batch_idx"]
    has_cell = hasattr(batch, "cell_offsets") and hasattr(batch, "lattice_matrix")

    def forces_from_pos(pos):
        """Frozen-feature direct forces with recomputed geometry."""
        src, dst = edge_index
        diff = pos[dst] - pos[src]
        if has_cell:
            offsets = batch.cell_offsets.to(dtype=pos.dtype)
            lat = batch.lattice_matrix.to(dtype=pos.dtype)[batch_idx[src]]
            diff = diff + torch.bmm(offsets.unsqueeze(1), lat).squeeze(1)
        dist = diff.norm(dim=-1).clamp(min=1e-8)

        node_z = cached.get("node_z")
        z_src = node_z[src] if node_z is not None else None
        z_dst = node_z[dst] if node_z is not None else None

        dE_dr = model.energy_head.compute_derivatives(
            edge_feats=cached["edge_feats"],
            edge_distances=dist,
            edge_rbf=cached["edge_rbf"],
            z_src=z_src,
            z_dst=z_dst,
        )
        r_hat = diff / dist.unsqueeze(-1)
        edge_f = dE_dr.unsqueeze(-1) * r_hat
        f = torch.zeros(N, 3, dtype=pos.dtype)
        f.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_f), edge_f)
        f.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_f), -edge_f)

        if model.use_angle_energy and triplet_indices.size(1) > 0:
            _, bond_dr, bond_morse, bond_beta = model.energy_head.compute_derivatives_with_dr(
                edge_feats=cached["edge_feats"],
                edge_distances=dist,
                edge_rbf=cached["edge_rbf"],
                z_src=z_src,
                z_dst=z_dst,
            )
            ej, ek = triplet_indices
            cos_vals = torch.cosine_similarity(diff[ej], diff[ek], dim=-1)
            f_angle = model.energy_head.compute_angle_forces(
                angle_feats=cached["angle_feats"],
                angle_values=cos_vals,
                triplet_index=triplet_indices,
                edge_index=edge_index,
                edge_distances=dist,
                diff=diff,
                num_atoms=N,
                bond_dr=bond_dr,
                bond_morse_factor=bond_morse,
                bond_beta=bond_beta,
                z_src=z_src,
                z_dst=z_dst,
            )
            f = f + f_angle

        return f.reshape(-1)

    dof = 3 * N
    H = torch.zeros(dof, dof)
    for col in range(dof):
        atom, coord = divmod(col, 3)
        pos_p = pos0.clone(); pos_p[atom, coord] += eps
        pos_m = pos0.clone(); pos_m[atom, coord] -= eps
        F_p = forces_from_pos(pos_p)
        F_m = forces_from_pos(pos_m)
        H[:, col] = -(F_p - F_m) / (2 * eps)

    return H


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SETUPS = [
    ("morse_no_angles",    dict(use_morse=True,  use_angles=False)),
    ("taylor_no_angles",   dict(use_morse=False, use_angles=False)),
    ("morse_with_angles",  dict(use_morse=True,  use_angles=True, use_stretch_bend=True)),
]


@pytest.mark.parametrize("name,kwargs", SETUPS)
def test_hessian_symmetry(name, kwargs):
    """H must be symmetric."""
    torch.manual_seed(0)
    model = make_model(**kwargs)
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [2.5, 0.0, 0.0],
                        [1.2, 2.1, 0.0]], dtype=torch.float32)
    batch = make_batch(pos)
    H = model.compute_analytical_hessian(batch).float()
    sym_err = (H - H.T).abs().max().item()
    print(f"[{name}] symmetry error = {sym_err:.2e}")
    assert sym_err < 1e-4, f"Hessian not symmetric: max|H-H^T| = {sym_err:.2e}"


@pytest.mark.parametrize("name,kwargs", SETUPS)
def test_acoustic_sum_rule(name, kwargs):
    """Each 3×3 block-row should sum to zero (translational invariance)."""
    torch.manual_seed(0)
    model = make_model(**kwargs)
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [2.5, 0.0, 0.0],
                        [1.2, 2.1, 0.0]], dtype=torch.float32)
    batch = make_batch(pos)
    N = pos.shape[0]
    H = model.compute_analytical_hessian(batch).float()

    for i in range(N):
        row_block = H[3*i:3*i+3, :].reshape(3, N, 3).sum(dim=1)  # (3, 3)
        err = row_block.abs().max().item()
        print(f"[{name}] atom {i} row-sum error = {err:.2e}")
        assert err < 1e-3, f"Acoustic sum rule violated for atom {i}: {err:.2e}"


@pytest.mark.parametrize("name,kwargs", SETUPS)
def test_numerical_consistency(name, kwargs):
    """H_analytical ≈ H_numerical_frozen (finite diff on frozen-GNN forces).

    Both use the same approximation: GNN features fixed at equilibrium,
    analytical energy differentiated w.r.t. positions.
    """
    torch.manual_seed(0)
    model = make_model(**kwargs)
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [2.5, 0.0, 0.0],
                        [1.2, 2.1, 0.0]], dtype=torch.float32)
    batch = make_batch(pos)

    H_ana = model.compute_analytical_hessian(batch).detach().float()
    H_num = numerical_hessian_frozen(model, batch, eps=1e-3)

    scale = H_num.abs().max().item()
    if scale < 1e-8:
        print(f"[{name}] Hessian is near-zero, skipping numerical check.")
        return

    rel_err = (H_ana - H_num).abs().max().item() / scale
    print(f"[{name}] max relative error vs numerical_frozen = {rel_err:.2e}  (scale={scale:.3f})")
    assert rel_err < 0.02, (
        f"Analytical Hessian deviates too much from frozen-feature numerical: {rel_err:.2%}"
    )


def test_angle_hessian_backbone_attachment_switch():
    """Angle-Hessian mode should optionally backprop into the angle encoder."""
    torch.manual_seed(0)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]],
        dtype=torch.float32,
    )
    batch = make_batch(pos)

    model_detached = make_model(
        use_morse=True,
        use_angles=True,
        use_stretch_bend=True,
        analytical_hessian_detach_angle_features=True,
    )
    loss_detached = model_detached.compute_analytical_hessian(batch).square().mean()
    loss_detached.backward()
    grad_detached = model_detached.angle_embedding[0].weight.grad

    model_attached = make_model(
        use_morse=True,
        use_angles=True,
        use_stretch_bend=True,
        analytical_hessian_detach_angle_features=False,
    )
    loss_attached = model_attached.compute_analytical_hessian(batch).square().mean()
    loss_attached.backward()
    grad_attached = model_attached.angle_embedding[0].weight.grad

    detached_norm = (
        0.0 if grad_detached is None else float(grad_detached.abs().max().item())
    )
    attached_norm = (
        0.0 if grad_attached is None else float(grad_attached.abs().max().item())
    )
    print(
        "angle-encoder grad norms:",
        {"detached": detached_norm, "attached": attached_norm},
    )
    assert attached_norm > detached_norm * 1.5


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Analytical Hessian sanity tests")
    print("=" * 60)
    failures = []
    for name, kwargs in SETUPS:
        for test_fn in [test_hessian_symmetry, test_acoustic_sum_rule, test_numerical_consistency]:
            label = f"{test_fn.__name__}[{name}]"
            try:
                test_fn(name, kwargs)
                print(f"  PASS  {label}")
            except AssertionError as e:
                print(f"  FAIL  {label}: {e}")
                failures.append(label)
            except Exception as e:
                print(f"  ERROR {label}: {e}")
                failures.append(label)

    try:
        test_angle_hessian_backbone_attachment_switch()
        print("  PASS  test_angle_hessian_backbone_attachment_switch")
    except AssertionError as e:
        print(f"  FAIL  test_angle_hessian_backbone_attachment_switch: {e}")
        failures.append("test_angle_hessian_backbone_attachment_switch")
    except Exception as e:
        print(f"  ERROR test_angle_hessian_backbone_attachment_switch: {e}")
        failures.append("test_angle_hessian_backbone_attachment_switch")

    print("=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
