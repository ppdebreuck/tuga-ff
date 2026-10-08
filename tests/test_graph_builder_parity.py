import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from tugaff.data.dataset import TugaGraphBuilder


def _reference_graph(structure, builder: TugaGraphBuilder):
    atomic_numbers = np.array(
        [
            site.element.Z if hasattr(site, "element") else site.Z
            for site in structure.species
        ],
        dtype=np.int64,
    )
    x = torch.tensor(atomic_numbers[:, None], dtype=torch.long)

    center_indices, neighbor_indices, images, distances = structure.get_neighbor_list(
        r=builder.cutoff
    )

    cart_coords = structure.cart_coords
    lattice_matrix = structure.lattice.matrix
    center_coords = cart_coords[center_indices]
    image_offsets = np.dot(images, lattice_matrix)
    neighbor_coords = cart_coords[neighbor_indices] + image_offsets
    diff_vecs = neighbor_coords - center_coords

    if builder.dynamic_cutoff:
        rc_bond = builder._element_bond_cutoffs_np
        z_src = x[center_indices].squeeze(-1).numpy()
        z_dst = x[neighbor_indices].squeeze(-1).numpy()
        rc_ij = 0.5 * (rc_bond[z_src] + rc_bond[z_dst])
        rc_mask = distances <= rc_ij
        center_indices = center_indices[rc_mask]
        neighbor_indices = neighbor_indices[rc_mask]
        diff_vecs = diff_vecs[rc_mask]
        distances = distances[rc_mask]
        images = images[rc_mask]

    norms = np.linalg.norm(diff_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    dir_vecs = diff_vecs / norms
    dir_x = np.round(dir_vecs[:, 0], decimals=6)
    dir_y = np.round(dir_vecs[:, 1], decimals=6)
    dir_z = np.round(dir_vecs[:, 2], decimals=6)
    dist_rounded = np.round(distances, decimals=6)

    sort_idx = np.lexsort((dir_z, dir_y, dir_x, dist_rounded, center_indices))
    center_indices = center_indices[sort_idx]
    neighbor_indices = neighbor_indices[sort_idx]
    diff_vecs = diff_vecs[sort_idx]
    distances = distances[sort_idx]
    images = images[sort_idx]

    counts = np.bincount(center_indices, minlength=len(structure))
    group_starts = np.zeros(len(counts), dtype=np.int64)
    group_starts[1:] = np.cumsum(counts)[:-1]
    starts_mapped = group_starts[center_indices]
    ranks = np.arange(len(center_indices)) - starts_mapped

    if builder.max_neighbors:
        mask = ranks < builder.max_neighbors
        center_indices = center_indices[mask]
        neighbor_indices = neighbor_indices[mask]
        diff_vecs = diff_vecs[mask]
        distances = distances[mask]
        images = images[mask]
        ranks = ranks[mask]

    edge_vecs = torch.tensor(diff_vecs, dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.tensor(center_indices, dtype=torch.long),
            torch.tensor(neighbor_indices, dtype=torch.long),
        ]
    )
    edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)

    src_indices = center_indices
    _, start_indices, triplet_counts = np.unique(
        src_indices, return_index=True, return_counts=True
    )
    valid_starts = start_indices[triplet_counts >= 2]
    valid_counts = triplet_counts[triplet_counts >= 2]

    triplets_left = []
    triplets_right = []
    rc_triplet_table = builder._element_triplet_cutoffs_np

    for start, count in zip(valid_starts, valid_counts):
        edges = np.arange(start, start + count)
        if builder.dynamic_cutoff:
            z_center = int(x[src_indices[start]].item())
            triplet_rc = float(rc_triplet_table[z_center])
            edges = edges[distances[edges] <= triplet_rc]
            if len(edges) < 2:
                continue
        if builder.max_neighbors_triplet is not None:
            edges = edges[ranks[edges] < builder.max_neighbors_triplet]
            if len(edges) < 2:
                continue

        edge_count = len(edges)
        e_left = np.repeat(edges, edge_count)
        e_right = np.tile(edges, edge_count)
        mask = e_left != e_right
        triplets_left.append(e_left[mask])
        triplets_right.append(e_right[mask])

    if triplets_left:
        t_left = np.concatenate(triplets_left)
        t_right = np.concatenate(triplets_right)
        triplet_index = torch.tensor(np.stack([t_left, t_right]), dtype=torch.long)
        angle_attr = torch.cosine_similarity(
            edge_vecs[triplet_index[0]],
            edge_vecs[triplet_index[1]],
        ).unsqueeze(-1)
    else:
        triplet_index = torch.empty((2, 0), dtype=torch.long)
        angle_attr = torch.empty((0, 1), dtype=torch.float32)

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "triplet_index": triplet_index,
        "angle_attr": angle_attr,
        "cell_offsets": torch.tensor(images, dtype=torch.long),
    }


def _make_structure():
    lattice = Lattice.cubic(5.43)
    return Structure(
        lattice,
        ["Si", "Si", "O"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0]],
    )


def _assert_graph_matches_reference(builder: TugaGraphBuilder):
    structure = _make_structure()
    graph = builder.get_graph(structure)
    reference = _reference_graph(structure, builder)

    assert torch.equal(graph.edge_index, reference["edge_index"])
    assert torch.equal(graph.triplet_index, reference["triplet_index"])
    assert torch.equal(graph.cell_offsets, reference["cell_offsets"])
    assert torch.allclose(graph.edge_attr, reference["edge_attr"])
    assert torch.allclose(graph.angle_attr, reference["angle_attr"])


def test_graph_builder_parity_fixed_cutoff():
    builder = TugaGraphBuilder(
        cutoff=6.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
        triplet_cutoff=4.0,
        dynamic_cutoff=False,
    )
    _assert_graph_matches_reference(builder)


def test_graph_builder_parity_dynamic_cutoff():
    builder = TugaGraphBuilder(
        cutoff=6.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
        triplet_cutoff=4.0,
        dynamic_cutoff=True,
    )
    _assert_graph_matches_reference(builder)


def _assert_ase_path_matches_pymatgen(builder: TugaGraphBuilder):
    """The ASE-native fast path must produce a graph identical to get_graph()."""
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = _make_structure()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    g_pmg = builder.get_graph(structure)
    g_ase = builder.get_graph_from_atoms(atoms)

    assert torch.equal(g_ase.edge_index, g_pmg.edge_index)
    assert torch.equal(g_ase.triplet_index, g_pmg.triplet_index)
    assert torch.equal(g_ase.cell_offsets, g_pmg.cell_offsets)
    assert torch.allclose(g_ase.edge_attr, g_pmg.edge_attr, atol=1e-5)
    assert torch.allclose(g_ase.angle_attr, g_pmg.angle_attr, atol=1e-5)


def test_graph_builder_ase_path_parity_fixed_cutoff():
    builder = TugaGraphBuilder(
        cutoff=6.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
        triplet_cutoff=4.0,
        dynamic_cutoff=False,
    )
    _assert_ase_path_matches_pymatgen(builder)


def test_graph_builder_ase_path_parity_dynamic_cutoff():
    builder = TugaGraphBuilder(
        cutoff=6.0,
        max_neighbors=8,
        max_neighbors_triplet=4,
        triplet_cutoff=4.0,
        dynamic_cutoff=True,
    )
    _assert_ase_path_matches_pymatgen(builder)
