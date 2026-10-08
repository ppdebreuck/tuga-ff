import logging
import os

import torch

logger = logging.getLogger(__name__)


def configure_precision(
    deterministic: bool,
    device,
    float32_matmul_precision: str = "highest",
) -> None:
    """Configure PyTorch float32 matmul policy and determinism process-globally.

    ``float32_matmul_precision`` follows PyTorch's native ``high``/``highest``
    terminology. ``high`` may use TF32 on CUDA; it does not promise a specific
    backend kernel.

    ``deterministic`` controls algorithm selection only. It does not override
    the requested matmul precision or any model tensor dtype.
    """
    if float32_matmul_precision not in {"high", "highest"}:
        raise ValueError(
            "float32_matmul_precision must be 'high' or 'highest'."
        )
    device_str = str(device).lower() if device is not None else ""
    is_cuda = "cuda" in device_str or (not device_str and torch.cuda.is_available())

    if not deterministic:
        torch.use_deterministic_algorithms(False)
        torch.set_float32_matmul_precision(float32_matmul_precision)
        if is_cuda:
            allow_tf32 = float32_matmul_precision == "high"
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            tf32_state = "enabled" if allow_tf32 else "disabled"
            msg = (
                "TugaFF-Precision: Active performance profile. "
                f"float32_matmul_precision='{float32_matmul_precision}', "
                f"TF32 {tf32_state}."
            )
        else:
            msg = (
                "TugaFF-Precision: Active performance profile. "
                f"float32_matmul_precision='{float32_matmul_precision}'."
            )
        logger.info(msg)
        print(msg)
    else:
        # Must be present before deterministic cuBLAS kernels are first used.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision(float32_matmul_precision)
        if is_cuda:
            allow_tf32 = float32_matmul_precision == "high"
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            tf32_state = "enabled" if allow_tf32 else "disabled"
            msg = (
                "TugaFF-Precision: Active deterministic algorithms. "
                f"float32_matmul_precision='{float32_matmul_precision}', "
                f"TF32 {tf32_state}, CuDNN deterministic."
            )
        else:
            msg = (
                "TugaFF-Precision: Active deterministic algorithms. "
                f"float32_matmul_precision='{float32_matmul_precision}'."
            )
        logger.info(msg)
        print(msg)
