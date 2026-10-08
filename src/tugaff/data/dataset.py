from typing import Dict, List, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from pymatgen.core import Structure
from torch_geometric.data import Data

class CrystalGraph(Data):
    """
    PyG Data object representing a crystal structure as a graph.

    This is a hierarchical graph with:
    - **Nodes**: Atoms in the crystal
    - **Edges**: Bonds between atoms (based on Voronoi decomposition)
    - **Triplets**: Angle relationships between pairs of edges sharing a node
      (used for 3-body interactions in the line graph)

    Attributes:
        x (Tensor): Node features, shape (N, F) where N=num_atoms.
                    Currently contains atomic numbers as indices.
        edge_index (Tensor): Edge connectivity, shape (2, E) where E=num_edges.
                             Format: [src_nodes, dst_nodes].
        edge_attr (Tensor): Edge features, shape (E, 1). Contains bond distances.
        triplet_index (Tensor): Triplet (angle) connectivity, shape (2, T).
                                Each column [i, j] means edges i and j share a node.
        angle_attr (Tensor): Angle features, shape (T, 1). Contains cos(θ) values.
        lattice_params (Tensor): Unit cell parameters, shape (1, 6).
                                 Contains [a, b, c, α, β, γ].
        y (Tensor, optional): Target property, shape varies.
        mat_id (str, optional): Material ID for tracking.
    """

    def __inc__(self, key, value, *args, **kwargs):
        """
        Defines how indices should be incremented during batching.

        PyG batches multiple graphs by concatenating node/edge tensors.
        Index tensors must be offset to point to the correct nodes/edges
        in the concatenated tensor.
        """
        if key == "edge_index":
            return self.x.size(0)  # Offset by number of nodes
        if key == "triplet_index":
            return self.edge_attr.size(0)  # Offset by number of edges
        if key == "cell_offsets":
            return 0  # No increment, these are relative integer shifts
        if key == "lattice_matrix":
            return 0  # (B, 3, 3) managed by batching
        return super().__inc__(key, value, *args, **kwargs)


class TugaGraphBuilder:
    """
    Converts pymatgen Structure objects to CrystalGraph objects.
    """

    def __init__(
        self,
        cutoff: float = 8.0,
        max_neighbors: int | None = 12,
        max_neighbors_triplet: int | None = 12,
        triplet_cutoff: float = 4.0,
        dynamic_cutoff: bool = False,
        element_cutoff_file: str | None = None,
        element_bond_cutoffs: torch.Tensor | None = None,
        element_triplet_cutoffs: torch.Tensor | None = None,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.max_neighbors_triplet = max_neighbors_triplet
        self.triplet_cutoff = triplet_cutoff

        # Element-cutoff mode
        self.dynamic_cutoff = dynamic_cutoff
        self.element_bond_cutoffs = element_bond_cutoffs
        self.element_triplet_cutoffs = element_triplet_cutoffs
        self._element_bond_cutoffs_np = None
        self._element_triplet_cutoffs_np = None

        if self.dynamic_cutoff:
            if (
                element_cutoff_file is not None
                and (self.element_bond_cutoffs is None or self.element_triplet_cutoffs is None)
            ):
                from ..utils.elements import load_cutoff_tables

                bond_cutoffs, triplet_cutoffs = load_cutoff_tables(element_cutoff_file)
                if self.element_bond_cutoffs is None:
                    self.element_bond_cutoffs = bond_cutoffs
                if self.element_triplet_cutoffs is None:
                    self.element_triplet_cutoffs = triplet_cutoffs

            if self.element_bond_cutoffs is None:
                from ..utils.elements import ELEMENT_BOND_CUTOFFS

                self._element_bond_cutoffs_np = ELEMENT_BOND_CUTOFFS.cpu().numpy()
            else:
                self._element_bond_cutoffs_np = (
                    self.element_bond_cutoffs.detach().cpu().numpy()
                )

            if self.element_triplet_cutoffs is None:
                from ..utils.elements import ELEMENT_TRIPLET_CUTOFFS

                self._element_triplet_cutoffs_np = (
                    ELEMENT_TRIPLET_CUTOFFS.cpu().numpy()
                )
            else:
                self._element_triplet_cutoffs_np = (
                    self.element_triplet_cutoffs.detach().cpu().numpy()
                )

    def get_graph(self, structure, properties=None, mat_id=None) -> "CrystalGraph":
        """
        Convert a pymatgen Structure to a CrystalGraph.

        Args:
            structure: pymatgen Structure object.
            properties: Optional dict of properties to attach (e.g., {"y": 1.5}).
            mat_id: Optional material ID string for tracking.

        Returns:
            CrystalGraph object ready for model input.
        """
        # 1. Node features (Atomic Numbers)
        atomic_numbers = np.fromiter(
            (
                site.element.Z
                if hasattr(site, "element")
                else site.Z
                for site in structure.species
            ),
            dtype=np.int64,
            count=len(structure),
        )

        # 2. Edges (bonds) with Cutoff + Max Neighbors strategy
        center_indices, neighbor_indices, images, distances = (
            structure.get_neighbor_list(r=self.cutoff)
        )

        # Calculate difference vectors manually
        cart_coords = structure.cart_coords
        lattice_matrix = structure.lattice.matrix

        # Vectorized calculations images are (N, 3), lattice is (3, 3)
        center_coords = cart_coords[center_indices]
        image_offsets = np.dot(images, lattice_matrix)
        neighbor_coords = cart_coords[neighbor_indices] + image_offsets
        diff_vecs = neighbor_coords - center_coords

        lat = structure.lattice
        lattice_params = np.array(
            [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma], dtype=np.float32
        )
        return self._assemble_graph(
            atomic_numbers=atomic_numbers,
            cart_coords=np.asarray(cart_coords, dtype=np.float32),
            lattice_matrix=np.asarray(lattice_matrix, dtype=np.float32),
            center_indices=np.asarray(center_indices, dtype=np.int64),
            neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
            images=np.asarray(images),
            distances=np.asarray(distances),
            diff_vecs=diff_vecs,
            lattice_params=lattice_params,
            volume=float(structure.volume),
            properties=properties,
            mat_id=mat_id,
        )

    def get_graph_from_atoms(self, atoms, properties=None, mat_id=None) -> "CrystalGraph":
        """Fast inference path: build a CrystalGraph directly from an ASE Atoms object.

        Skips the pymatgen ``Structure`` round-trip and its per-site Python overhead
        by calling pymatgen's C++ neighbor search (``find_points_in_spheres``, the
        same routine ``Structure.get_neighbor_list`` wraps) straight on the ASE
        coordinate arrays. The downstream sorting / triplet / tensor assembly is
        identical to :meth:`get_graph`, so the resulting graph (edges, triplets,
        cell offsets, ordering) matches the pymatgen path exactly.
        """
        from pymatgen.optimization.neighbors import find_points_in_spheres

        atomic_numbers = atoms.get_atomic_numbers().astype(np.int64)
        cart_coords = np.ascontiguousarray(atoms.get_positions(), dtype=float)
        lattice_matrix = np.ascontiguousarray(atoms.cell.array, dtype=float)
        pbc = np.ascontiguousarray(atoms.pbc, dtype=int)

        center_indices, neighbor_indices, images, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=float(self.cutoff), pbc=pbc,
            lattice=lattice_matrix, tol=1e-8,
        )
        # Drop self-pairs (i == j at zero offset, distance 0) to match
        # Structure.get_neighbor_list, which excludes the central atom.
        nonself = distances > 1e-8
        center_indices = center_indices[nonself]
        neighbor_indices = neighbor_indices[nonself]
        images = images[nonself]
        distances = distances[nonself]

        # diff_vec = r_j + image·lattice − r_i (periodic-correct, matches get_graph).
        # np.dot on a contiguous copy avoids a spurious numpy matmul warning that
        # the C-routine's non-contiguous offset array can trigger.
        image_offsets = np.dot(np.ascontiguousarray(images, dtype=float), lattice_matrix)
        diff_vecs = (
            cart_coords[neighbor_indices] + image_offsets - cart_coords[center_indices]
        )

        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        lattice_params = np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)
        return self._assemble_graph(
            atomic_numbers=atomic_numbers,
            cart_coords=cart_coords.astype(np.float32),
            lattice_matrix=lattice_matrix.astype(np.float32),
            center_indices=center_indices.astype(np.int64),
            neighbor_indices=neighbor_indices.astype(np.int64),
            images=np.rint(images).astype(np.int64),
            distances=distances,
            diff_vecs=diff_vecs,
            lattice_params=lattice_params,
            volume=float(atoms.get_volume()),
            properties=properties,
            mat_id=mat_id,
        )

    def _assemble_graph(
        self,
        *,
        atomic_numbers: np.ndarray,
        cart_coords: np.ndarray,
        lattice_matrix: np.ndarray,
        center_indices: np.ndarray,
        neighbor_indices: np.ndarray,
        images: np.ndarray,
        distances: np.ndarray,
        diff_vecs: np.ndarray,
        lattice_params: np.ndarray,
        volume: float,
        properties=None,
        mat_id=None,
    ) -> "CrystalGraph":
        """Shared sorting + triplet + tensor assembly for both graph entry points."""
        num_sites = len(atomic_numbers)
        x = torch.from_numpy(atomic_numbers[:, None])

        # Apply sorting with deterministic tie-breaking for degenerate shells.
        # Without tie-breaking, neighbors at the same distance can be selected
        # differently in unit cell vs supercell, causing the GNN to produce
        # different features for identical local environments. This breaks
        # phonon calculations.
        # 1. Apply element-specific cutoff filtering (Physical Boundary)
        if getattr(self, "dynamic_cutoff", False):
            rc_bond = self._element_bond_cutoffs_np
            z_src = atomic_numbers[center_indices]
            z_dst = atomic_numbers[neighbor_indices]
            rc_ij = 0.5 * (rc_bond[z_src] + rc_bond[z_dst])

            rc_mask = distances <= rc_ij
            center_indices = center_indices[rc_mask]
            neighbor_indices = neighbor_indices[rc_mask]
            diff_vecs = diff_vecs[rc_mask]
            distances = distances[rc_mask]
            images = images[rc_mask]

        # 2. Sort and Create ranks
        norms = np.linalg.norm(diff_vecs, axis=1)
        norms = np.where(norms < 1e-10, 1.0, norms)
        dir_vecs = diff_vecs / norms[:, None]
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

        counts = np.bincount(center_indices, minlength=num_sites)
        group_starts = np.zeros(len(counts), dtype=np.int64)
        group_starts[1:] = np.cumsum(counts)[:-1]
        starts_mapped = group_starts[center_indices]
        ranks = np.arange(len(center_indices)) - starts_mapped

        # 3. Filter by max_neighbors (Memory Guardrail)
        if self.max_neighbors:
            mask = ranks < self.max_neighbors
            center_indices = center_indices[mask]
            neighbor_indices = neighbor_indices[mask]
            diff_vecs = diff_vecs[mask]
            distances = distances[mask]
            images = images[mask]
            ranks = ranks[mask]

        src_indices = center_indices
        dst_indices = neighbor_indices
        edge_vecs = torch.from_numpy(diff_vecs.astype(np.float32, copy=False))
        edge_ranks = torch.from_numpy(ranks.astype(np.int64, copy=False))
        edge_dists = torch.from_numpy(distances.astype(np.float32, copy=False))
        edge_index = torch.from_numpy(
            np.stack([src_indices, dst_indices]).astype(np.int64, copy=False)
        )

        # Edge features: bond distances, shape (E, 1)
        edge_attr = edge_dists.unsqueeze(-1)

        # 3. Triplets (for 3-body / angle interactions)
        #
        # The graph stores a broad candidate set of local triplets. The effective
        # angular support is decided smoothly inside the energy head from the two
        # leg distances, so hard triplet membership should not be the main physics
        # selector for the PES.

        # Vectorized triplet construction. src_indices is sorted ascending (the
        # lexsort above uses center_indices as the primary key), so each atom's
        # edges form a contiguous block. We compute a per-edge "is a valid triplet
        # leg" mask, then emit all ordered same-atom leg pairs (a != b) in the exact
        # group/row-major order the previous Python loop produced.
        keep = np.ones(len(src_indices), dtype=bool)
        if getattr(self, "dynamic_cutoff", False):
            rc_triplet_table = self._element_triplet_cutoffs_np
            # x is atomic_numbers[:, None]; the center of edge e is src_indices[e].
            rc_tri_per_edge = rc_triplet_table[atomic_numbers[src_indices]]
            keep &= distances <= rc_tri_per_edge
        if self.max_neighbors_triplet is not None:
            keep &= ranks < self.max_neighbors_triplet

        kept_idx = np.nonzero(keep)[0]  # global edge indices, ascending
        if kept_idx.size > 0:
            # Per-kept-edge center atom (non-decreasing); group into per-atom legs.
            g = src_indices[kept_idx]
            _, first, gcounts = np.unique(g, return_index=True, return_counts=True)
            sel = gcounts >= 2
            sizes = gcounts[sel]
            bases = first[sel]  # offset of each group's first leg within kept_idx
        else:
            sizes = np.empty(0, dtype=np.int64)

        if sizes.size == 0:
            triplet_index = torch.empty((2, 0), dtype=torch.long)
            angle_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            sq = sizes * sizes
            grp = np.repeat(np.arange(sizes.size), sq)  # group id per c*c cell
            within = np.arange(sq.sum()) - np.repeat(np.cumsum(sq) - sq, sq)
            c_of = sizes[grp]
            a = within // c_of  # leg index (outer) within group -> matches np.repeat
            b = within % c_of   # leg index (inner) within group -> matches np.tile
            nondiag = a != b
            base = bases[grp]
            t_left = kept_idx[base + a][nondiag]
            t_right = kept_idx[base + b][nondiag]

            triplet_index = torch.from_numpy(np.stack([t_left, t_right]))
            vec_j = edge_vecs[triplet_index[0]]
            vec_k = edge_vecs[triplet_index[1]]
            cos_theta = torch.cosine_similarity(vec_j, vec_k)
            angle_attr = cos_theta.unsqueeze(-1)


        # 4. Lattice parameters: (a, b, c, α, β, γ)
        # Shape (1, 6) so PyG batching stacks to (B, 6)
        lattice_params_t = torch.from_numpy(
            np.asarray(lattice_params, dtype=np.float32).reshape(1, 6)
        )
        lattice_matrix_t = torch.from_numpy(
            np.asarray(lattice_matrix, dtype=np.float32)
        ).unsqueeze(0)  # (1, 3, 3)

        # 5. Build the graph object
        pos = torch.from_numpy(np.asarray(cart_coords, dtype=np.float32))
        cell_offsets = torch.from_numpy(images.astype(np.int64, copy=False))  # (E, 3)

        graph = CrystalGraph(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_offsets=cell_offsets,
            triplet_index=triplet_index,
            angle_attr=angle_attr,
            lattice_params=lattice_params_t,
            lattice_matrix=lattice_matrix_t,
        )

        # Always set mat_id so PyG collation doesn't fail on mixed batches
        graph.mat_id = mat_id or ""

        if properties:
            for k, v in properties.items():
                if k == "y":
                    # Handle Vector vs Scalar targets
                    val = torch.tensor(v, dtype=torch.float32)
                    if val.ndim == 0:
                        val = val.unsqueeze(0)  # Scalar -> (1,)
                    graph.y = val
                else:
                    setattr(graph, k, v)

        # Attach volume for stress calculation
        graph.volume = torch.tensor([volume], dtype=torch.float32)

        return graph

    def _process_batch(self, batch_data):
        """Helper to process a batch of structures in one go."""
        return [self.get_graph(s, p, m) for s, p, m in batch_data]

    def get_graphs(
        self,
        structures: List[Structure],
        properties: Optional[List[Optional[Dict]]] = None,
        mat_ids: Optional[List[Optional[str]]] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> List["CrystalGraph"]:
        """
        Convert a list of pymatgen Structures to CrystalGraph objects in parallel.

        Args:
            structures: List of pymatgen Structure objects.
            properties: Optional list of property dicts corresponding to each structure.
                        If None, all structures will have no properties attached.
            mat_ids: Optional list of material IDs corresponding to each structure.
            n_jobs: Number of parallel jobs. Follows joblib conventions:
                    - 1 (default): Sequential processing (no parallelism overhead).
                    - -1: Use all available CPUs.
                    - Positive integer: Use exactly n_jobs workers.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of CrystalGraph objects.
        """
        n = len(structures)

        # Handle None inputs by creating lists of None
        if properties is None:
            properties = [None] * n
        if mat_ids is None:
            mat_ids = [None] * n

        if len(properties) != n:
            raise ValueError(
                f"Length mismatch: {n} structures vs {len(properties)} properties"
            )
        if len(mat_ids) != n:
            raise ValueError(
                f"Length mismatch: {n} structures vs {len(mat_ids)} mat_ids"
            )

        if n_jobs == 1:
            # Sequential processing with optional progress bar
            iterable = zip(structures, properties, mat_ids)
            if show_progress:
                from tqdm import tqdm

                iterable = tqdm(
                    list(iterable),
                    desc="Building graphs (Serial)",
                    unit="struct",
                )
            return [self.get_graph(s, props, mid) for s, props, mid in iterable]

        # Parallel processing with Manual Batching

        # Determine effective n_jobs
        if n_jobs < 0:
            import multiprocessing

            n_jobs = multiprocessing.cpu_count()

        # Heuristic for chunk size based on benchmarks
        base_chunk = n // (n_jobs * 4)
        chunk_size = min(50, max(10, base_chunk))

        # Generator for chunks
        batches = []
        for i in range(0, n, chunk_size):
            batch = list(
                zip(
                    structures[i : i + chunk_size],
                    properties[i : i + chunk_size],
                    mat_ids[i : i + chunk_size],
                )
            )
            batches.append(batch)

        jobs = (delayed(self._process_batch)(batch) for batch in batches)

        if show_progress:
            from tqdm import tqdm

            # Use generator to preserve order and assemble efficiently
            parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")

            graphs = []
            with tqdm(
                total=n, desc=f"Building graphs ({n_jobs} workers)", unit="struct"
            ) as pbar:
                for batch_result in parallel(jobs):
                    graphs.extend(batch_result)
                    pbar.update(len(batch_result))
        else:
            batch_results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
            graphs = [g for batch in batch_results for g in batch]

        return graphs
