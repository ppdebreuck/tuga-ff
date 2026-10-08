# TugaFF

> [!WARNING]
> TugaFF is still under active development — expect breaking changes.

## Quickstart

The `tuga-ff` package (calculator + checkpoint downloader) is published as
wheels at `https://alexandria.icams.rub.de/data/tff/`. `pip`/`uv` resolves the
latest version from that index automatically:

```bash
uv pip install tuga-ff --find-links https://alexandria.icams.rub.de/data/tff/
```

```python
from ase.build import bulk
from tugaff.calculator import TugaFFCalculator

atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
atoms.calc = TugaFFCalculator()  # auto-downloads + caches the latest checkpoint
print(atoms.get_potential_energy())
```

## Calculator

The ASE calculator is the main integration point for downstream use. It lives in
[`src/tugaff/calculator.py`](src/tugaff/calculator.py) and exposes
`TugaFFCalculator`, which:

- resolves and (optionally) auto-downloads a registered checkpoint by name, or loads a local `.ckpt`
- rebuilds the graph on every call to match training-time behavior (no cached state)
- predicts total energy, forces, and stress
- supports CPU or CUDA execution
- supports `autograd` and `direct` force modes
- optionally JIT-compiles the model via `torch.compile`

```python
from ase.build import bulk
from tugaff.calculator import TugaFFCalculator

atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)

calc = TugaFFCalculator(
    model_name="v1-mp",            # auto-downloads + caches the checkpoint ("v1-mp", "v1-matpes-3d", "v1-matpes-all")
    device="cpu",
    force_mode="autograd",         # "autograd" (default) or "direct"
    precision_profile="fast",      # "fast" (default), "balanced", or "precise"
    compile_model=True,
)
atoms.calc = calc

# First call carries a one-time JIT compile cost; subsequent calls are fast.
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
print(f"Potential Energy: {energy:.6f} eV")
```

You can also load a local checkpoint directly with `model_path="path/to/model.ckpt"`,
which overrides `model_name`.

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `"v1-mp"` | Registered checkpoint name to download and cache (`"v1-mp"`, `"v1-matpes-3d"`, `"v1-matpes-all"`). |
| `model_path` | `str` or `Path` | `None` | Local path to a `.ckpt`. Overrides `model_name`. |
| `device` | `str` | `"cuda"` if available | Target device: `"cpu"` or `"cuda"`. |
| `force_mode` | `str` | `"autograd"` | Force mode: `"autograd"` (reverse-mode autograd) or `"direct"` (pairwise Morse forces). |
| `precision_profile` | `str` | `"fast"` | Precision profile: `"fast"`, `"balanced"`, or `"precise"`. |
| `compile_model` | `bool` | `True` | JIT-compile submodules via `torch.compile`. |
| `cache_dir` | `str` or `Path` | `None` | Custom directory for weights and inductor/triton compile caches. |
| `max_force` | `float` or `None` | `50.0` | Maximum force norm clipping threshold for stability. |

### Checkpoint downloader & cache

If `model_path` is not given (or a registered name is passed), the calculator
resolves, downloads, and caches the checkpoint:

- **Default model**: `"v1-mp"`
- **Available models**: `"v1-mp"`, `"v1-matpes-3d"`, `"v1-matpes-all"`
- **Registry host**: `https://alexandria.icams.rub.de/data/tff_checkpoints/`
- **Local cache**: `~/.cache/tuga-ff/` by default; override with `cache_dir=` or the
  `TUGAFF_CACHE_DIR` environment variable. When a custom cache dir is set, `torch.compile`
  artifacts are stored under `<cache_dir>/inductor` and `<cache_dir>/triton`.

The registry lives in [`src/tugaff/checkpoints.py`](src/tugaff/checkpoints.py).

### Force modes

- `force_mode="autograd"` (default): exact forces/stress via reverse-mode autograd. Works for all heads.
- `force_mode="direct"`: analytical pairwise Morse forces; bypasses autograd graph construction (only if the checkpoint has a pairwise direct force head).

### TF32 on Ampere+ GPUs

On modern NVIDIA GPUs the calculator auto-enables TF32 tensor cores
(`torch.set_float32_matmul_precision("high")`), unless a
matmul precision was already configured.


## Architecture

- `src/tugaff/models/`: NN layers, graph transformer backbone, Taylor energy head, Lightning module
- `src/tugaff/data/`: graph construction, dataset loading, baked/chunked helpers, structure conversion
- `src/tugaff/calculator.py`: ASE calculator wrapper
- `src/tugaff/checkpoints.py`: hosted checkpoint registry + cached downloader
- `tests/`: unit tests for models, heads, features, and calculator integration

## Installation

See [Quickstart](#quickstart) above to install the published package. To work
on this repo from source:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .        # basic
uv pip install -e ".[dev]" # + dev tools (pytest)
```
