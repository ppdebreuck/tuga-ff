"""Checkpoint registry helpers.

Checkpoints are hosted publicly on alexandria under
`https://alexandria.icams.rub.de/data/tff_checkpoints/` and registered here so
the calculator can fetch them by name. To publish a new model, follow
`notes/PUBLISH_CHECKPOINT.md`.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

DEFAULT_MODEL_NAME = "v1-mp"

_CHECKPOINT_BASE_URL = "https://alexandria.icams.rub.de/data/tff_checkpoints"

AVAILABLE_CHECKPOINTS: dict[str, dict[str, str]] = {
    "v1-mp": {
        "filename": "tuga-ff-v1-mp.ckpt",
        "description": (
            "TugaFF v1 force field trained on the Materials Project dataset."
        ),
        "url": f"{_CHECKPOINT_BASE_URL}/tuga-ff-v1-mp.ckpt",
    },
    "v1-matpes-3d": {
        "filename": "tuga-ff-v1-matpes-3d.ckpt",
        "description": (
            "TugaFF v1 force field trained on MatPES (3D) data."
        ),
        "url": f"{_CHECKPOINT_BASE_URL}/tuga-ff-v1-matpes-3d.ckpt",
    },
    "v1-matpes-all": {
        "filename": "tuga-ff-v1-matpes-all.ckpt",
        "description": (
            "TugaFF v1 force field trained on the full MatPES dataset."
        ),
        "url": f"{_CHECKPOINT_BASE_URL}/tuga-ff-v1-matpes-all.ckpt",
    },
}


def available_model_names() -> list[str]:
    return sorted(AVAILABLE_CHECKPOINTS)


def checkpoint_root(root: str | os.PathLike[str] | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser()
    env_root = os.environ.get("TUGAFF_CACHE_DIR")
    if env_root:
        return Path(env_root).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "tuga-ff"
    return Path.home() / ".cache" / "tuga-ff"


def checkpoint_path(name: str, root: str | os.PathLike[str] | None = None) -> Path:
    if name not in AVAILABLE_CHECKPOINTS:
        available = ", ".join(available_model_names()) or "none"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return checkpoint_root(root) / AVAILABLE_CHECKPOINTS[name]["filename"]


def download_checkpoint(
    name: str = DEFAULT_MODEL_NAME,
    root: str | os.PathLike[str] | None = None,
    *,
    verbose: bool = True,
) -> Path:
    if name not in AVAILABLE_CHECKPOINTS:
        available = ", ".join(available_model_names()) or "none"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")

    metadata = AVAILABLE_CHECKPOINTS[name]
    if not metadata.get("url"):
        raise RuntimeError(
            f"Checkpoint '{name}' is registered but not linked to a download URL yet. "
            "Pass model_path=... to load a local checkpoint."
        )
    target = checkpoint_path(name, root)
    target.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"TugaFF: downloading {name}")
        print(f"  URL: {metadata['url']}")
        print(f"  Cache: {target}")

    with urllib.request.urlopen(metadata["url"]) as response:
        total = int(response.headers.get("Content-Length", 0))
        with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as tmp_file:
            tmp_path = Path(tmp_file.name)
            progress = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {name}",
                disable=not verbose,
            )
            try:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    progress.update(len(chunk))
            finally:
                progress.close()
    tmp_path.replace(target)
    return target


def ensure_checkpoint(
    name: str = DEFAULT_MODEL_NAME,
    root: str | os.PathLike[str] | None = None,
    *,
    verbose: bool = True,
) -> Path:
    target = checkpoint_path(name, root)
    if target.exists():
        if verbose:
            print(f"TugaFF: using cached {name} at {target}")
        return target
    return download_checkpoint(name=name, root=root, verbose=verbose)
