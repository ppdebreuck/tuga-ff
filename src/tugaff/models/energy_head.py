"""Energy readout heads for Tuga-FF."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool

from .layers import MLP


# MP2020 elemental reference energies used by the BU architecture.  Keeping the
# values in-package makes initialization reproducible on training and inference
# machines without importing matbench-discovery at model construction time.
MP_ELEMENTAL_REFERENCE_ENERGIES = {
    1: -3.392726045,
    2: -0.00905951,
    3: -1.9089228666666667,
    4: -3.739412865,
    5: -6.679391770833334,
    6: -9.2286654925,
    7: -8.336494925,
    8: -4.94668871125,
    9: -1.9114789675,
    10: -0.02593678,
    11: -1.3225252934482759,
    12: -1.60028005,
    13: -3.74557583,
    14: -5.42531803,
    15: -5.413302506666667,
    16: -4.136449866875,
    17: -1.84853666,
    18: -0.06880822,
    19: -1.110398947,
    20: -2.00559988,
    21: -6.332469105,
    22: -7.895492016666666,
    23: -9.08390607,
    24: -9.65304747,
    25: -9.162015292068965,
    26: -8.47002121,
    27: -7.108317795,
    28: -5.78013668,
    29: -4.09920667,
    30: -1.2597436100000001,
    31: -3.0280960225,
    32: -4.623027855,
    33: -4.659118405,
    34: -3.49591147765625,
    35: -1.55302833,
    36: -0.05671467,
    37: -0.9805340725,
    38: -1.6894934533333332,
    39: -6.466471113333333,
    40: -8.54770063,
    41: -10.10130504,
    42: -10.84565011,
    43: -10.360638945,
    44: -9.27440254,
    45: -7.36430787,
    46: -5.17988181,
    47: -2.8325560033333335,
    48: -0.92288976,
    49: -2.75168373,
    50: -4.009571855,
    51: -4.12900124,
    52: -3.1433058933333338,
    53: -1.47336635,
    54: -0.03617417,
    55: -0.8954023720689656,
    56: -1.91897494,
    57: -4.936007105,
    58: -5.933089155,
    59: -4.780905755,
    60: -4.7681474325,
    61: -4.7505423225,
    62: -4.718586135,
    63: -10.2570018,
    64: -14.07612224,
    65: -4.6343661,
    66: -4.60678684,
    67: -4.58240887,
    68: -4.56771881,
    69: -4.475835423333334,
    70: -1.5396082800000002,
    71: -4.52095052,
    72: -9.95718903,
    73: -11.85777763,
    74: -12.95813023,
    75: -12.444527185,
    76: -11.22736743,
    77: -8.83843418,
    78: -6.07113332,
    79: -3.273882,
    80: -0.303680365,
    81: -2.3626431466666666,
    82: -3.71264707,
    83: -3.84048913,
    89: -4.1211750075,
    90: -7.41385825,
    91: -9.51466466,
    92: -11.29141001,
    93: -12.94777968125,
    94: -14.26783833,
}


class SumPoolingEnergyHead(nn.Module):
    """Per-node energy MLP followed by graph-wise sum pooling.

    The head predicts an extensive total energy for each graph:

        E_graph = sum_i MLP(node_i) + species_shift(Z_i)

    It intentionally has no Morse, Taylor, angle, pair-table, or analytic-force
    branches. Forces and stress are obtained only by autograd through the full
    model.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        num_atom_types: int = 100,
        activation: str = "silu",
        num_mlp_layers: int = 2,
        species_shift_init: str = "zeros",
    ) -> None:
        super().__init__()
        d_hidden = d_hidden or d_model
        self.node_energy_mlp = MLP(
            d_in=d_model,
            d_hidden=d_hidden,
            d_out=1,
            num_layers=num_mlp_layers,
            activation=activation,
        )
        self.species_shift = nn.Embedding(num_atom_types + 1, 1)
        nn.init.zeros_(self.species_shift.weight)
        if species_shift_init == "mp_elemental_ref":
            with torch.no_grad():
                for atomic_number, energy in MP_ELEMENTAL_REFERENCE_ENERGIES.items():
                    if atomic_number <= num_atom_types:
                        self.species_shift.weight[atomic_number, 0] = energy
        elif species_shift_init != "zeros":
            raise ValueError(
                "species_shift_init must be 'zeros' or 'mp_elemental_ref', "
                f"got {species_shift_init!r}."
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_z: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        node_z = node_z.view(-1).long()
        node_energy = self.node_energy_mlp(node_feats) + self.species_shift(node_z)
        # A calculator call contains exactly one graph.  A plain reduction is
        # equivalent to scatter/index_add in that case and avoids a known
        # Inductor+autograd miscompile of the single-segment scatter.  Training
        # batches still use normal graph-wise pooling.  Both branches compile.
        if num_graphs == 1:
            energy = node_energy.sum(dim=0, keepdim=True)
        else:
            energy = global_add_pool(node_energy, batch_idx, size=num_graphs)
        return energy.view(-1)


# Backward-compatible import name for older user code. This is not a Taylor
# implementation; deprecated arguments are intentionally unsupported.
TaylorEnergyHead = SumPoolingEnergyHead
