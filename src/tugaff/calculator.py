"""ASE calculator for simplified Tuga-FF checkpoints."""

from __future__ import annotations

import os
import warnings
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.stress import full_3x3_to_voigt_6_stress
from torch_geometric.data import Batch

from .data.dataset import TugaGraphBuilder
from .models.lit_ff_model import LitTugaForceField


class TugaFFCalculator(Calculator):
    """ASE calculator backed by a `LitTugaForceField` checkpoint."""

    implemented_properties = ["energy", "forces", "stress"]
    precision_profiles = {
        "fast": {
            "geometry_dtype": "fp32",
            "backbone_dtype": "auto",
            "head_dtype": "fp32",
            "float32_matmul_precision": "high",
        },
        "balanced": {
            "geometry_dtype": "fp32",
            "backbone_dtype": "float32",
            "head_dtype": "fp32",
            "float32_matmul_precision": "highest",
        },
        "precise": {
            "geometry_dtype": "fp64",
            "backbone_dtype": "fp64",
            "head_dtype": "fp64",
            "float32_matmul_precision": "highest",
        },
    }

    def __init__(
        self,
        model_name=None,
        model_path=None,
        model=None,
        device=None,
        force_mode="autograd",
        element_bias=None,
        max_neighbors="auto",
        max_neighbors_triplet="auto",
        cutoff="auto",
        triplet_cutoff="auto",
        precision_profile="fast",
        geometry_dtype=None,
        head_dtype=None,
        backbone_dtype=None,
        float32_matmul_precision=None,
        compile_model=True,
        cache_dir=None,
        deterministic: bool | None = None,
        max_force: float | None = 50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if force_mode not in {"autograd", "direct"}:
            raise ValueError("force_mode must be 'autograd' or 'direct'.")

        self.force_mode = force_mode
        if force_mode == "direct":
            self.implemented_properties = ["energy", "forces"]
        self.element_bias = element_bias or {}
        self.max_neighbors = max_neighbors
        self.max_neighbors_triplet = max_neighbors_triplet
        self.cutoff = cutoff
        self.triplet_cutoff = triplet_cutoff
        if precision_profile not in self.precision_profiles:
            choices = "', '".join(self.precision_profiles)
            raise ValueError(f"precision_profile must be one of '{choices}'.")
        profile = self.precision_profiles[precision_profile]
        geometry_dtype = geometry_dtype or profile["geometry_dtype"]
        backbone_dtype = backbone_dtype or profile["backbone_dtype"]
        head_dtype = head_dtype or profile["head_dtype"]
        float32_matmul_precision = (
            float32_matmul_precision or profile["float32_matmul_precision"]
        )
        self.precision_profile = precision_profile
        self.geometry_dtype = geometry_dtype
        self.head_dtype = head_dtype
        self.backbone_dtype = backbone_dtype
        self.float32_matmul_precision = float32_matmul_precision
        if geometry_dtype not in {"fp32", "fp64"}:
            raise ValueError("geometry_dtype must be 'fp32' or 'fp64'.")
        if head_dtype not in {"fp32", "fp64"}:
            raise ValueError("head_dtype must be 'fp32' or 'fp64'.")
        if backbone_dtype not in {"auto", "bf16", "float32", "fp16", "fp64"}:
            raise ValueError(
                "backbone_dtype must be 'auto', 'bf16', 'float32', 'fp16', "
                "or 'fp64'."
            )
        if float32_matmul_precision not in {"high", "highest"}:
            raise ValueError(
                "float32_matmul_precision must be 'high' or 'highest'."
            )
        self.deterministic = False if deterministic is None else bool(deterministic)
        self.compile_model = bool(compile_model)
        self.max_force = float(max_force) if max_force is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        supports_bf16 = self._hardware_precision_support()
        self._resolved_backbone_dtype = self._resolve_backbone_dtype(
            backbone_dtype,
            supports_bf16=supports_bf16,
        )
        self._head_dtype = (
            torch.float64
            if head_dtype == "fp64"
            else torch.float32
        )
        self._geometry_dtype = (
            torch.float64
            if geometry_dtype == "fp64"
            else torch.float32
        )
        self._amp_dtype = {
            "float32": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp64": None,
        }[self._resolved_backbone_dtype]
        if self.deterministic:
            self.compile_model = False

        from tugaff.utils.precision import configure_precision

        configure_precision(
            self.deterministic,
            self.device,
            float32_matmul_precision=self.float32_matmul_precision,
        )
        self.model_path = model_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self._float_batch_keys = None
        self._compiled_autograd_compiler = None
        self._compiled_autograd_dynamic = False

        effective_cache_dir = self.cache_dir or os.environ.get("TUGAFF_CACHE_DIR")
        if effective_cache_dir is not None:
            cache_path = Path(effective_cache_dir).expanduser().resolve()
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR", str(cache_path / "inductor")
            )
            os.environ.setdefault("TRITON_CACHE_DIR", str(cache_path / "triton"))

        if model is not None:
            self.model = model
            self._configure_loaded_model()
        else:
            if model_path is None:
                from .checkpoints import (
                    DEFAULT_MODEL_NAME,
                    available_model_names,
                    ensure_checkpoint,
                )

                name = model_name or DEFAULT_MODEL_NAME
                if name not in available_model_names():
                    raise ValueError(f"Unknown model name '{name}'.")
                model_path = str(ensure_checkpoint(name, root=self.cache_dir))
                self.model_path = model_path
                self.model_name = name
            self.load_model(model_path)

    def _hardware_precision_support(self) -> bool:
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("A CUDA device was requested, but CUDA is unavailable.")

        supports_bf16 = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                supports_bf16 = torch.cuda.is_bf16_supported(
                    including_emulation=False
                )
        return supports_bf16

    def _resolve_backbone_dtype(
        self,
        requested: str,
        *,
        supports_bf16: bool | None = None,
    ) -> str:
        device = torch.device(self.device)
        if supports_bf16 is None:
            supports_bf16 = self._hardware_precision_support()
        if requested == "auto":
            if supports_bf16:
                return "bf16"
            return "float32"
        if requested in {"bf16", "fp16"} and device.type != "cuda":
            raise ValueError(
                f"backbone_dtype='{requested}' requires a CUDA device."
            )
        if requested == "bf16" and not supports_bf16:
            raise ValueError(
                "backbone_dtype='bf16' requested, but this CUDA device does "
                "not have native BF16 support."
            )
        return requested

    def load_model(self, model_path):
        self.model = LitTugaForceField.load_from_checkpoint(
            model_path,
            map_location=self.device,
            strict=False,
        )
        self._configure_loaded_model()

    def _configure_loaded_model(self):
        # Mixed-precision backbones keep FP32 parameters and use autocast.
        # A precise backbone stores and evaluates all model parameters in FP64.
        backbone_parameter_dtype = (
            torch.float64
            if self._resolved_backbone_dtype == "fp64"
            else torch.float32
        )
        self.model.to(device=self.device, dtype=backbone_parameter_dtype)
        self.model.model._geometry_dtype = self._geometry_dtype
        self.model.model.energy_head.to(dtype=self._head_dtype)
        if self.model.model.direct_force_head is not None:
            self.model.model.direct_force_head.to(dtype=self._head_dtype)
        self.model.eval()
        self.model.freeze()

        if self.force_mode == "direct" and self.model.model.direct_force_head is None:
            raise ValueError(
                "force_mode='direct' requires a checkpoint with a direct_force_head."
            )

        hparams = self.model.hparams
        cutoff = (
            hparams.get("edge_rbf_stop", 8.0)
            if self.cutoff == "auto"
            else float(self.cutoff)
        )
        max_neighbors = (
            hparams.get("max_neighbors", 12)
            if self.max_neighbors == "auto"
            else (
                None
                if self.max_neighbors is None
                else int(self.max_neighbors)
            )
        )
        max_neighbors_triplet = (
            hparams.get("max_neighbors_triplet", 12)
            if self.max_neighbors_triplet == "auto"
            else (
                None
                if self.max_neighbors_triplet is None
                else int(self.max_neighbors_triplet)
            )
        )
        triplet_cutoff = (
            hparams.get("triplet_cutoff", 4.0)
            if self.triplet_cutoff == "auto"
            else float(self.triplet_cutoff)
        )
        dynamic_cutoff = bool(hparams.get("dynamic_cutoff", False))
        element_cutoff_file = hparams.get("element_cutoff_file", None)
        element_bond_cutoffs = getattr(self.model.model, "element_bond_cutoffs", None)
        element_triplet_cutoffs = getattr(
            self.model.model, "element_triplet_cutoffs", None
        )

        self.graph_builder = TugaGraphBuilder(
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            max_neighbors_triplet=max_neighbors_triplet,
            triplet_cutoff=triplet_cutoff,
            dynamic_cutoff=dynamic_cutoff,
            element_cutoff_file=element_cutoff_file,
            element_bond_cutoffs=element_bond_cutoffs,
            element_triplet_cutoffs=element_triplet_cutoffs,
        )
        self._setup_compilation()

        model_identifier = self.model_name or (
            Path(self.model_path).name if self.model_path else "provided"
        )
        cutoff_info = "dynamic cutoffs" if dynamic_cutoff else f"cutoff={cutoff}A"
        print(
            f"TugaFF-Calculator: loaded '{model_identifier}' on {self.device} "
            f"({cutoff_info}, precision_profile={self.precision_profile}, "
            f"geometry_dtype={self.geometry_dtype}, "
            f"backbone_dtype={self.backbone_dtype}"
            f"->{self._resolved_backbone_dtype}, head_dtype={self.head_dtype}"
            f", float32_matmul_precision={self.float32_matmul_precision}, "
            f"deterministic={self.deterministic})"
        )

    def _setup_compilation(self) -> None:
        if not self.compile_model or not hasattr(torch, "compile"):
            return
        if "mps" in str(self.device).lower():
            self.compile_model = False
            return
        if self.force_mode == "direct":
            self.model.model._compiled_direct = torch.compile(
                self.model.model._compute_direct_outputs,
                dynamic=True,
            )
        else:
            self.model.model._compiled_outputs = torch.compile(
                self.model.model._compute_outputs,
                dynamic=True,
            )

    def _cast_batch_floats(self, batch: Batch) -> Batch:
        if self._float_batch_keys is None:
            self._float_batch_keys = tuple(
                key
                for key in batch.keys()
                if isinstance(batch[key], torch.Tensor)
                and batch[key].is_floating_point()
                and batch[key].dtype != self._geometry_dtype
            )
        for key in self._float_batch_keys:
            batch[key] = batch[key].to(self._geometry_dtype)
        return batch

    def _run_model_forward(self, batch: Batch):
        amp = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if self._amp_dtype is not None
            else nullcontext()
        )
        with warnings.catch_warnings(), amp:
            # PyTorch 2.10 + CUDA 12.8 can reject the fused-bias cuBLASLt
            # kernel, emit this warning, and immediately retry successfully
            # through the unfused cuBLAS path. Keep the calculator CLI quiet
            # while allowing all non-recoverable cuBLAS errors to propagate.
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED "
                    r"when calling cublasLtMatmul.*"
                ),
                category=UserWarning,
            )
            return self.model.model(
                batch,
                compute_forces=True,
                compute_stress=self.force_mode == "autograd",
                force_mode=self.force_mode,
            )

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if self.force_mode == "direct" and "stress" in properties:
            raise PropertyNotImplementedError(
                "Stress is unavailable when force_mode='direct'."
            )
        super().calculate(atoms, properties, system_changes)
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        graph = self.graph_builder.get_graph_from_atoms(atoms)
        batch = Batch.from_data_list((graph,)).to(self.device)
        batch = self._cast_batch_floats(batch)

        grad_context = (
            torch.inference_mode()
            if self.force_mode == "direct"
            else torch.enable_grad()
        )
        with grad_context:
            energy_pred, force_pred, stress_pred = self._run_model_forward(batch)

        total_energy = float(energy_pred.detach().cpu().item())
        if self.element_bias:
            counts = Counter(atoms.get_chemical_symbols())
            for symbol, bias in self.element_bias.items():
                total_energy -= float(bias) * counts.get(symbol, 0)

        self.results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "forces": None,
            "stress": None,
        }

        if force_pred is not None:
            f = force_pred.detach().cpu().numpy()
            if self.max_force is not None:
                norms = np.linalg.norm(f, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.max_force / np.clip(norms, 1e-12, None))
                f = f * scale
            self.results["forces"] = f

        if stress_pred is not None:
            s_np = stress_pred.squeeze().detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(s_np)


TugaCalculator = TugaFFCalculator
