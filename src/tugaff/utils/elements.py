"""Per-element dynamic cutoff tables used by graph construction.

The cutoffs are stored in ``element_cutoffs.json`` so they can be regenerated
from analysis runs without rewriting Python logic. The graph builder uses:

- bond cutoff: pair mixing, ``Rc_ij = 0.5 * (Rc_i + Rc_j)``
- triplet cutoff: center-element cutoff, ``Rc_triplet(center)``

If the JSON file is unavailable for any reason, we fall back to conservative
defaults matching the previous behavior closely enough to keep baking usable.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MAX_ATOMIC_NUMBER = 118
DEFAULT_BOND_CUTOFF = 5.0
DEFAULT_TRIPLET_CUTOFF = 3.0

_DEFAULT_DATA_PATH = Path(__file__).with_name("element_cutoffs.json")


def _fallback_tables() -> tuple[torch.Tensor, torch.Tensor]:
    bond = torch.full((MAX_ATOMIC_NUMBER + 1,), DEFAULT_BOND_CUTOFF, dtype=torch.float32)
    triplet = torch.full((MAX_ATOMIC_NUMBER + 1,), DEFAULT_TRIPLET_CUTOFF, dtype=torch.float32)

    # Match the old coarse heuristic as a safe backup.
    organics = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    alkali = [3, 11, 19, 37, 55]

    for z in organics:
        if z <= MAX_ATOMIC_NUMBER:
            bond[z] = 3.5
            triplet[z] = 2.5

    for z in alkali:
        if z <= MAX_ATOMIC_NUMBER:
            bond[z] = 6.0
            triplet[z] = 4.0

    return bond, triplet


def _resolve_data_path(path: str | Path | None) -> Path:
    if path is None:
        return _DEFAULT_DATA_PATH
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return Path(__file__).resolve().parent / candidate


def load_cutoff_tables(path: str | Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    bond, triplet = _fallback_tables()
    try:
        payload = json.loads(_resolve_data_path(path).read_text())
        elements = payload.get("elements")
        if isinstance(elements, list):
            for entry in elements:
                if not isinstance(entry, dict):
                    continue
                z = int(entry["z"])
                if 0 <= z <= MAX_ATOMIC_NUMBER:
                    if "bond_cutoff" in entry:
                        bond[z] = float(entry["bond_cutoff"])
                    if "triplet_cutoff" in entry:
                        triplet[z] = float(entry["triplet_cutoff"])
        else:
            for key, value in payload.get("bond_cutoffs", {}).items():
                z = int(key)
                if 0 <= z <= MAX_ATOMIC_NUMBER:
                    bond[z] = float(value)
            for key, value in payload.get("triplet_cutoffs", {}).items():
                z = int(key)
                if 0 <= z <= MAX_ATOMIC_NUMBER:
                    triplet[z] = float(value)
    except Exception:
        # Keep the fallback tables if the data file cannot be read.
        pass
    return bond, triplet


ELEMENT_BOND_CUTOFFS, ELEMENT_TRIPLET_CUTOFFS = load_cutoff_tables()
