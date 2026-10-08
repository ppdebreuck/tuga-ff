import math

import torch
import torch.nn as nn


def _broadcast_cutoff_params(
    distances: torch.Tensor, cutoff: float | torch.Tensor, width: float | torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast cutoff/width to match ``distances`` while preserving scalar behavior."""
    cutoff_t = torch.as_tensor(cutoff, dtype=distances.dtype, device=distances.device)
    cutoff_t = torch.broadcast_to(cutoff_t, distances.shape)

    if width is None:
        width_t = cutoff_t
    else:
        width_t = torch.as_tensor(width, dtype=distances.dtype, device=distances.device)
        width_t = torch.broadcast_to(width_t, distances.shape)
    return cutoff_t, width_t


def get_atom_features(species):
    """
    Get atomic number for an atom.

    Args:
        species (pymatgen.core.Species or Element): The atomic species.

    Returns:
        torch.Tensor: Tensor containing atomic number (1-based index).
    """
    if hasattr(species, "element"):
        element = species.element
    else:
        element = species

    # Return atomic number (Z) directly
    # Can be used with nn.Embedding(num_embeddings=100)
    return torch.tensor([element.Z], dtype=torch.long)


def _c4_smoothstep_complement(x: torch.Tensor) -> torch.Tensor:
    """Degree-9 switch from 1 to 0 with zero derivatives through order four."""
    x2 = x * x
    x4 = x2 * x2
    x5 = x4 * x
    left = 1.0 - x5 * (
        126.0 - 420.0 * x + 540.0 * x2 - 315.0 * x2 * x + 70.0 * x4
    )

    y = 1.0 - x
    y2 = y * y
    y4 = y2 * y2
    y5 = y4 * y
    right = y5 * (
        126.0 - 420.0 * y + 540.0 * y2 - 315.0 * y2 * y + 70.0 * y4
    )
    return torch.where(x <= 0.5, left, right)


def _c4_smoothstep_complement_derivative(x: torch.Tensor) -> torch.Tensor:
    x2 = x * x
    x4 = x2 * x2
    y = 1.0 - x
    y2 = y * y
    y4 = y2 * y2
    return -630.0 * x4 * y4


def _c4_smoothstep_complement_second_derivative(x: torch.Tensor) -> torch.Tensor:
    x2 = x * x
    x3 = x2 * x
    y = 1.0 - x
    y2 = y * y
    y3 = y2 * y
    return -2520.0 * x3 * y3 * (y - x)


def smooth_polynomial_cutoff(
    distances: torch.Tensor, cutoff: float | torch.Tensor, width: float | torch.Tensor | None = None
) -> torch.Tensor:
    """
    C4 polynomial smoothstep cutoff with exact zero support at and beyond the cutoff.

    When ``width`` is omitted, the switch spans the full interval ``[0, cutoff]``.
    When provided, support stays at 1 below ``cutoff - width`` and decays smoothly
    to 0 over the final ``width`` Angstroms before the cutoff.
    """
    cutoff_t, width_t = _broadcast_cutoff_params(distances, cutoff, width)
    hard_mask = width_t <= 0
    safe_width = torch.where(hard_mask, torch.ones_like(width_t), width_t)
    inner = cutoff_t - safe_width
    x = ((distances - inner) / safe_width).clamp(min=0.0, max=1.0)
    switch = _c4_smoothstep_complement(x)

    ones = torch.ones_like(distances)
    zeros = torch.zeros_like(distances)
    weights = torch.where(distances < inner, ones, switch)
    weights = torch.where(distances >= cutoff_t, zeros, weights)
    return torch.where(
        hard_mask,
        torch.where(distances < cutoff_t, ones, zeros),
        weights,
    )


def smooth_polynomial_cutoff_with_derivative(
    distances: torch.Tensor, cutoff: float | torch.Tensor, width: float | torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the C4 cutoff and its first distance derivative."""
    cutoff_t, width_t = _broadcast_cutoff_params(distances, cutoff, width)
    hard_mask = width_t <= 0
    safe_width = torch.where(hard_mask, torch.ones_like(width_t), width_t)
    inner = cutoff_t - safe_width
    x = ((distances - inner) / safe_width).clamp(min=0.0, max=1.0)

    zeros = torch.zeros_like(distances)
    ones = torch.ones_like(distances)
    switch = _c4_smoothstep_complement(x)
    weights = torch.where(distances < inner, ones, switch)
    weights = torch.where(distances >= cutoff_t, zeros, weights)

    dswitch_dx = _c4_smoothstep_complement_derivative(x)
    dswitch_dr = dswitch_dx / safe_width
    dweights = torch.where(distances < inner, zeros, dswitch_dr)
    dweights = torch.where(distances >= cutoff_t, zeros, dweights)
    hard_weights = torch.where(distances < cutoff_t, ones, zeros)
    return torch.where(hard_mask, hard_weights, weights), torch.where(
        hard_mask, zeros, dweights
    )


def smooth_polynomial_cutoff_with_second_derivative(
    distances: torch.Tensor, cutoff: float | torch.Tensor, width: float | torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the C4 cutoff and its first/second distance derivatives."""
    weights, dweights = smooth_polynomial_cutoff_with_derivative(
        distances, cutoff=cutoff, width=width
    )
    cutoff_t, width_t = _broadcast_cutoff_params(distances, cutoff, width)
    hard_mask = width_t <= 0
    safe_width = torch.where(hard_mask, torch.ones_like(width_t), width_t)
    inner = cutoff_t - safe_width
    x = ((distances - inner) / safe_width).clamp(min=0.0, max=1.0)
    zeros = torch.zeros_like(distances)
    d2switch_dx2 = _c4_smoothstep_complement_second_derivative(x)
    d2switch_dr2 = d2switch_dx2 / (safe_width**2)
    d2weights = torch.where(distances < inner, zeros, d2switch_dr2)
    d2weights = torch.where(distances >= cutoff_t, zeros, d2weights)
    return weights, dweights, torch.where(hard_mask, zeros, d2weights)


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Backward-compatible alias for the shared C4 polynomial cutoff.
    """
    return smooth_polynomial_cutoff(distances, cutoff=cutoff, width=cutoff)


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, sigma2=None):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if sigma2 is None:
            # Heuristic: match the spacing
            step = offset[1] - offset[0]
            sigma2 = 0.5 * (step) ** 2
        self.register_buffer("offset", offset)
        self.register_buffer(
            "limit", torch.as_tensor(sigma2)
        )  # Use as_tensor to avoid copy warning

    def forward(self, dist):
        # dist: (N, ) or (N, 1)
        # offset: (G, )
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        diff = dist - self.offset
        return torch.exp(-(diff**2) / (2 * self.limit))


# ---------------------------------------------------------------------------
# Bessel radial basis functions (smoother alternative to Gaussian RBF)
# ---------------------------------------------------------------------------


class BesselBasis(nn.Module):
    """
    Bessel radial basis functions (first kind, zeroth order).

    φ_n(r) = sqrt(2/r_cut) * sin(n * π * r / r_cut) / r

    These form an orthogonal basis on [0, r_cut] and smoothly go to zero
    at the cutoff, making them ideal for smooth edge embeddings.

    Advantages over Gaussian RBFs:
        - Orthogonal: each basis captures independent radial information
        - Natural cutoff: φ_n(r_cut) = 0 without a separate envelope
        - Fewer bases needed: 8 Bessel ≈ 80 Gaussian in expressiveness
    """

    def __init__(self, num_basis: int = 8, cutoff: float = 8.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Frequencies: n * pi / r_cut for n = 1..num_basis
        freq = torch.arange(1, num_basis + 1, dtype=torch.float32) * math.pi / cutoff
        self.register_buffer("freq", freq)

        # Normalization factor
        self.register_buffer("norm", torch.as_tensor(math.sqrt(2.0 / cutoff)))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (E,) or (E, 1) edge distances.

        Returns:
            (E, num_basis) Bessel basis values.
        """
        if dist.dim() == 2:
            dist = dist.squeeze(-1)

        # Clamp to avoid div-by-zero at r=0
        dist_safe = dist.clamp(min=1e-8)

        # (E, num_basis)
        return (
            self.norm
            * torch.sin(dist_safe.unsqueeze(-1) * self.freq)
            / dist_safe.unsqueeze(-1)
        )


class AngleGaussianBasis(nn.Module):
    """Gaussian basis on cos(theta) with centers placed uniformly in theta.

    A plain ``GaussianSmearing(-1, 1)`` is uniform in cos(theta), so its angular
    resolution is *coarsest* near 90 deg (cos=0, where d theta / d cos is largest)
    -- exactly where most equilibrium bond angles sit. Placing the centers at
    cos(theta_grid) for theta_grid uniform in [0, pi] gives uniform-in-angle
    resolution, sharpening the bending force constants k_theta that the soft
    transverse-acoustic / FC3 modes depend on.

    Per-center width tracks the local cos-spacing (d cos = sin(theta) d theta),
    floored so the near-linear / near-overlap ends do not become spiky.
    """

    def __init__(self, num_gaussians: int = 96, sin_floor: float = 0.3):
        super().__init__()
        theta = torch.linspace(0.0, math.pi, num_gaussians)
        offset = torch.cos(theta)  # (G,) decreasing 1 -> -1
        dtheta = math.pi / max(num_gaussians - 1, 1)
        local_step = torch.sin(theta).clamp(min=sin_floor) * dtheta  # cos-space spacing
        sigma2 = 0.5 * local_step**2 + 1e-6
        self.register_buffer("offset", offset)
        self.register_buffer("sigma2", sigma2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        diff = x - self.offset
        return torch.exp(-(diff**2) / (2 * self.sigma2))
