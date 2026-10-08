from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward: FFN(x) = (Swish(xW1) ⊙ xW3) W2"""

    def __init__(self, d_model: int, dim_feedforward: int, bias: bool = False):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * dim_feedforward, bias=bias)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(value) * gate)


class AttentionSupport(NamedTuple):
    """Precomputed support tensors shared by every interaction layer."""

    value: torch.Tensor
    positive: torch.Tensor
    log_prior: torch.Tensor


def prepare_attention_support(
    support: torch.Tensor, dtype: torch.dtype | None = None
) -> AttentionSupport:
    """Cast support and cache its mask/log prior once per graph level."""
    value = support if dtype is None else support.to(dtype=dtype)
    positive = value > 0
    safe = value.clamp_min(1e-12)
    log_prior = torch.where(
        positive,
        torch.log(safe),
        torch.full_like(value, -1e9),
    )
    return AttentionSupport(value=value, positive=positive, log_prior=log_prior)


class MLP(nn.Sequential):
    """Multi-Layer Perceptron."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        activation: str = "silu",
        dropout: float = 0.0,
        keep_dropout_layer: bool = False,
    ):
        layers = []
        in_dim = d_in
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(self._get_activation(activation))
            if dropout > 0.0 or keep_dropout_layer:
                # Dropout(p=0) is an exact identity; keep_dropout_layer preserves
                # checkpoint indices for modules trained with dropout enabled.
                layers.append(nn.Dropout(dropout))
            in_dim = d_hidden

        layers.append(nn.Linear(d_hidden, d_out))
        super().__init__(*layers)

    def _get_activation(self, activation: str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()


def _support_weighted_softmax_impl(
    scores: torch.Tensor,
    index: torch.Tensor,
    support: torch.Tensor | AttentionSupport | None,
    num_nodes: int | None = None,
    mode: str = "weighted",
) -> torch.Tensor:
    """Grouped weighted softmax with exact zero output for zero-support groups."""
    if support is None:
        return pyg_softmax(scores, index, num_nodes=num_nodes)

    if scores.numel() == 0:
        return scores

    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1 if index.numel() > 0 else 0

    cached = (
        support
        if isinstance(support, AttentionSupport)
        else prepare_attention_support(support, scores.dtype)
    )
    view_support = cached.value
    while view_support.dim() < scores.dim():
        view_support = view_support.unsqueeze(-1)

    positive = cached.positive
    while positive.dim() < scores.dim():
        positive = positive.unsqueeze(-1)
    # Fall through to the main, safe, division-guarded path for all modes including 'mask'.
    # This avoids division-by-zero NaNs when a node has zero active neighbors (isolated atoms).
    masked_scores = torch.where(
        positive, scores, torch.full_like(scores, float("-inf"))
    )

    if num_nodes == 1:
        # All edges belong to the same destination group. Plain reductions are
        # equivalent here and avoid an Inductor CUDA miscompile of the
        # single-segment scatter reductions used by periodic one-atom cells.
        grouped_max = masked_scores.amax(dim=0, keepdim=True)
        gathered_max = grouped_max.expand_as(scores)
    else:
        grouped_max = torch.full(
            (num_nodes,) + scores.shape[1:],
            float("-inf"),
            dtype=scores.dtype,
            device=scores.device,
        )
        expand_shape = (-1,) + (1,) * (scores.dim() - 1)
        expanded_index = index.view(*expand_shape).expand_as(scores)
        grouped_max.scatter_reduce_(
            0, expanded_index, masked_scores, reduce="amax", include_self=True
        )
        gathered_max = grouped_max.index_select(0, index)
    valid = positive & torch.isfinite(gathered_max)

    stabilized_scores = torch.where(
        valid, scores - gathered_max, torch.zeros_like(scores)
    )
    if mode == "smooth_split":
        # Use a softer support in attention normalization so disappearing
        # neighbors leave the softmax partition smoothly without making their
        # message contribution scale as s^2 once the physical message gate is
        # applied below.
        support_weight = view_support.clamp_min(0.0).sqrt().to(dtype=scores.dtype)
    elif mode in {"weighted", "smooth_softmax"}:
        support_weight = view_support.to(dtype=scores.dtype)
    else:
        support_weight = torch.ones_like(view_support, dtype=scores.dtype)
    exp_terms = torch.where(
        valid,
        torch.exp(stabilized_scores) * support_weight,
        torch.zeros_like(scores),
    )

    if num_nodes == 1:
        grouped_denom = exp_terms.sum(dim=0, keepdim=True)
        gathered_denom = grouped_denom.expand_as(scores)
    else:
        grouped_denom = torch.zeros_like(grouped_max)
        grouped_denom.index_add_(0, index, exp_terms)
        gathered_denom = grouped_denom.index_select(0, index)

    weights = torch.zeros_like(scores)
    nonzero_denom = gathered_denom > 0
    weights = torch.where(
        valid & nonzero_denom,
        exp_terms / gathered_denom.clamp_min(1e-7),
        weights,
    )
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


class GraphTransformerLayer(nn.Module):
    """
    Transformer layer for graph data.
    Pre-norm attention + pre-norm FFN with residual connections.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_edge: int,
        dim_feedforward: int = 512,
        activation: str = "relu",
        dropout: float = 0.0,
        use_swiglu: bool = False,
        support_mode: str = "weighted",
        d_rbf: int | None = None,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.norm1 = RMSNorm(d_model)
        # Fused Q/K/V projection avoids three launches in the attention hot path.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # One GEMM produces value bias/value gamma/key gamma/key beta. The gamma
        # and key-beta blocks start at zero, matching identity FiLM conditioning.
        self.edge_condition_proj = nn.Linear(d_edge, 4 * d_model, bias=False)
        with torch.no_grad():
            self.edge_condition_proj.weight[d_model:].zero_()

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-Head Radial/Distance Gate & Radial Bias
        if d_rbf is None or d_rbf <= 0:
            raise ValueError("v4 attention requires a positive radial-basis width.")
        self.d_rbf = int(d_rbf)
        hidden_dim = max(d_model // 2, 32)
        self.radial_hidden_proj = nn.Linear(self.d_rbf, 2 * hidden_dim)
        self.radial_gate_out = nn.Linear(hidden_dim, nhead)
        self.radial_bias_out = nn.Linear(hidden_dim, nhead)
        nn.init.constant_(self.radial_gate_out.bias, 1.0)
        nn.init.zeros_(self.radial_bias_out.weight)
        nn.init.zeros_(self.radial_bias_out.bias)

        self.norm2 = RMSNorm(d_model)
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.support_mode = support_mode

        if use_swiglu:
            self.ffn = SwiGLU(d_model, dim_feedforward)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )

    def _get_activation(self, activation: str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU()
        return nn.ReLU()

    def _support_weighted_attention(
        self,
        x_norm,
        edge_index,
        edge_attr,
        support: AttentionSupport,
        raw_rbf,
    ) -> torch.Tensor:
        num_nodes = x_norm.size(0)
        src, dst = edge_index

        # Single fused GEMM, then split into per-head q/k/v. The fused weight is
        # laid out as cat([Wq, Wk, Wv], dim=0), so reshaping the output to
        # (N, 3, nhead, head_dim) and unbinding dim 1 recovers q, k, v exactly.
        qkv = self.qkv_proj(x_norm).view(num_nodes, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(1)

        conditioned = self.edge_condition_proj(edge_attr).view(
            -1, 4, self.nhead, self.head_dim
        )
        edge_v = conditioned[:, 0]
        value_gamma, key_gamma = torch.tanh(conditioned[:, 1:3]).unbind(1)
        key_beta = conditioned[:, 3]
        k_modulated = k[src] * (1.0 + key_gamma) + key_beta

        semantic_compat = (q[dst] * k_modulated).sum(dim=-1) * self.scale

        radial_hidden = F.silu(self.radial_hidden_proj(raw_rbf))
        gate_hidden, bias_hidden = radial_hidden.chunk(2, dim=-1)
        gate = torch.sigmoid(self.radial_gate_out(gate_hidden))
        bias = self.radial_bias_out(bias_hidden)
        logits = gate * semantic_compat + bias
        logits = logits + support.log_prior.view(-1, 1)

        alpha = _support_weighted_softmax_impl(
            logits,
            dst,
            support,
            num_nodes=num_nodes,
            mode="mask",
        )
        alpha = self.dropout(alpha)

        values = (v[src] + edge_v) * (1.0 + value_gamma)
        messages = values * alpha.unsqueeze(-1)
        messages = messages * support.value.view(-1, 1, 1)

        if num_nodes == 1:
            out = messages.sum(dim=0, keepdim=True)
        else:
            out = messages.new_zeros((num_nodes, self.nhead, self.head_dim))
            out.index_add_(0, dst, messages)
        return self.out_proj(out.reshape(num_nodes, self.d_model))

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        support: AttentionSupport,
        raw_rbf,
    ):
        residual = x
        x_norm = self.norm1(x)
        x_attn = self._support_weighted_attention(
            x_norm, edge_index, edge_attr, support, raw_rbf=raw_rbf
        )
        x = residual + x_attn

        # Pre-Norm FFN
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn

        return x


class InteractionBlock(nn.Module):
    """Stack of InteractionLayers: hierarchical message passing."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        d_triplet: int | None = None,
        num_layers: int = 1,
        activation: str = "relu",
        use_swiglu: bool = False,
        dropout: float = 0.0,
        support_mode: str = "weighted",
        num_edge_basis: int | None = None,
        num_angle_basis: int | None = None,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
        edge_update_initial_scale: float = 0.1,
        angle_update_initial_scale: float = 0.0,
        normalize_angle_update: bool = False,
    ):
        super().__init__()
        self.support_mode = support_mode
        d_triplet = d_triplet or d_model

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                InteractionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    d_triplet=d_triplet,
                    activation=activation,
                    use_swiglu=use_swiglu,
                    dropout=dropout,
                    support_mode=support_mode,
                    num_edge_basis=num_edge_basis,
                    num_angle_basis=num_angle_basis,
                    update_hidden_dim=update_hidden_dim,
                    nonlinear_updates=nonlinear_updates,
                    edge_update_initial_scale=edge_update_initial_scale,
                    angle_update_initial_scale=angle_update_initial_scale,
                    normalize_angle_update=normalize_angle_update,
                )
            )

    def forward(
        self,
        node_feats,
        edge_index,
        edge_feats,
        edge_support: AttentionSupport,
        triplet_indices,
        angle_feats,
        triplet_support: AttentionSupport,
        edge_rbf,
        angle_rbf,
    ):
        x = node_feats
        e = edge_feats
        a = angle_feats

        for layer in self.layers:
            x, e, a = layer(
                x,
                edge_index,
                e,
                edge_support,
                triplet_indices,
                a,
                triplet_support,
                edge_rbf,
                angle_rbf,
            )

        return x, e


class SymmetricNonlinearUpdate(nn.Module):
    """Compact nonlinear update from a target and an unordered source pair.

    A shared nonlinear source encoder followed by sum and product pooling is
    permutation invariant in ``(left, right)``.  Target, sum, and product
    features are fused only at the narrow ``hidden_dim`` width. Sources are
    encoded once before pair indexing, so a line graph never materializes two
    full-width bond tensors per triplet. Additive projections avoid allocating
    a concatenated ``[num_pairs, 3 * hidden_dim]`` tensor.

    The returned message is the complete update candidate; there is no linear
    fallback or interpolation path.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        hidden_dim = min(int(hidden_dim), source_dim, target_dim)
        self.source_norm = RMSNorm(source_dim)
        self.source_in = nn.Linear(source_dim, hidden_dim, bias=False)
        self.source_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_norm = RMSNorm(target_dim)
        self.target_proj = nn.Linear(target_dim, hidden_dim, bias=False)

        # This additive decomposition is exactly as expressive as one linear
        # map over cat([target, pair_sum, pair_product]), with a lower peak.
        self.target_gate = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.sum_gate = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.product_gate = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.context_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_norm = RMSNorm(hidden_dim)
        self.message_out = nn.Linear(hidden_dim, target_dim, bias=False)
        self.message_gate = nn.Linear(hidden_dim, 1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Initialize update keys when strictly loading an older checkpoint."""
        legacy_key = prefix + "context_gate.weight"
        if legacy_key in state_dict:
            legacy_weight = state_dict.pop(legacy_key)
            chunks = legacy_weight.chunk(3, dim=1)
            for name, value in zip(
                ("target_gate.weight", "sum_gate.weight", "product_gate.weight"),
                chunks,
            ):
                state_dict.setdefault(prefix + name, value)

        # Reinitialize incompatible update tensors rather than failing an
        # otherwise useful backbone checkpoint load.
        for name, value in self.state_dict().items():
            key = prefix + name
            loaded = state_dict.get(key)
            if loaded is None or loaded.shape != value.shape:
                state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        target: torch.Tensor,
        sources: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        encoded_sources = self.source_out(
            F.silu(self.source_in(self.source_norm(sources)))
        )
        left, right = pair_index
        encoded_left = encoded_sources[left]
        encoded_right = encoded_sources[right]
        pair_sum = (encoded_left + encoded_right) * (2.0**-0.5)
        pair_product = encoded_left * encoded_right
        target_context = self.target_proj(self.target_norm(target))

        fused = self.target_gate(target_context)
        fused = fused + self.sum_gate(pair_sum)
        fused = fused + self.product_gate(pair_product)
        value, gate = fused.chunk(2, dim=-1)
        context = self.context_norm(self.context_out(F.silu(value) * gate))
        conditional_gate = 2.0 * torch.sigmoid(self.message_gate(context))
        return conditional_gate * self.message_out(context)


class NodeToEdgeAdapter(nn.Module):
    """Symmetric feedback from evolving atom states into bonds.

    ``nonlinear_updates=False`` restores the original checkpoint-compatible
    ``RMSNorm -> Linear -> symmetric sum`` update exactly.
    """

    def __init__(
        self,
        d_model: int,
        support_mode: str = "weighted",
        initial_scale: float = 0.1,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
    ) -> None:
        super().__init__()
        self.support_mode = support_mode
        self.nonlinear_updates = bool(nonlinear_updates)
        self.scale = nn.Parameter(torch.full((d_model,), float(initial_scale)))
        self.edge_norm = RMSNorm(d_model)
        if self.nonlinear_updates:
            self.update = SymmetricNonlinearUpdate(
                source_dim=d_model,
                target_dim=d_model,
                hidden_dim=update_hidden_dim,
            )
        else:
            self.node_norm = RMSNorm(d_model)
            self.proj = nn.Linear(d_model, d_model, bias=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load the selected branch while ignoring weights from the other."""
        if not self.nonlinear_updates:
            for key in list(state_dict):
                if key.startswith(prefix + "mixer.") or key.startswith(
                    prefix + "update."
                ):
                    state_dict.pop(key)
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        for key in list(state_dict):
            if key.startswith(prefix + "mixer."):
                suffix = key[len(prefix + "mixer.") :]
                if suffix != "mix_logit":
                    state_dict.setdefault(prefix + "update." + suffix, state_dict[key])
                state_dict.pop(key)
        state_dict.pop(prefix + "node_norm.weight", None)
        state_dict.pop(prefix + "proj.weight", None)
        state_dict.setdefault(prefix + "scale", self.scale.detach().clone())
        state_dict.setdefault(
            prefix + "edge_norm.weight", self.edge_norm.weight.detach().clone()
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        support: AttentionSupport,
    ) -> torch.Tensor:
        if self.nonlinear_updates:
            node_message = self.update(
                target=e,
                sources=x,
                pair_index=edge_index,
            )
        else:
            src, dst = edge_index
            projected = self.proj(self.node_norm(x))
            node_message = (projected[src] + projected[dst]) * (2.0**-0.5)
        residual_support = (
            support.positive.to(dtype=e.dtype)
            if self.support_mode == "mask"
            else support.value
        )
        return self.edge_norm(
            e + residual_support.unsqueeze(-1) * self.scale * node_message
        )


class SymmetricAngleUpdate(nn.Module):
    """Update angles from an unordered pair of bond states.

    The outer scale is configurable: legacy models leave it at zero, while
    from-scratch nonlinear models can activate it immediately. Optional output
    normalization matches the BU residual-update pattern. ``nonlinear_updates=False``
    restores the original checkpoint-compatible linear symmetric update.
    """

    def __init__(
        self,
        d_model: int,
        d_triplet: int,
        support_mode: str = "weighted",
        initial_scale: float = 0.0,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
        normalize_residual: bool = False,
    ) -> None:
        super().__init__()
        self.support_mode = support_mode
        self.nonlinear_updates = bool(nonlinear_updates)
        self.normalize_residual = bool(normalize_residual)
        self.scale = nn.Parameter(torch.full((d_triplet,), float(initial_scale)))
        if self.nonlinear_updates and self.normalize_residual:
            self.angle_norm = RMSNorm(d_triplet)
        if self.nonlinear_updates:
            self.update = SymmetricNonlinearUpdate(
                source_dim=d_model,
                target_dim=d_triplet,
                hidden_dim=update_hidden_dim,
            )
        else:
            self.edge_norm = RMSNorm(d_model)
            self.proj = nn.Linear(d_model, d_triplet, bias=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load the selected branch while ignoring weights from the other."""
        if not self.nonlinear_updates:
            for key in list(state_dict):
                if key.startswith(prefix + "mixer.") or key.startswith(
                    prefix + "update."
                ):
                    state_dict.pop(key)
            # A checkpoint from before angle feedback must remain an exact
            # no-op when loaded strictly into the linear branch.
            for name, value in self.state_dict().items():
                state_dict.setdefault(prefix + name, value.detach().clone())
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        for key in list(state_dict):
            if key.startswith(prefix + "mixer."):
                suffix = key[len(prefix + "mixer.") :]
                if suffix != "mix_logit":
                    state_dict.setdefault(prefix + "update." + suffix, state_dict[key])
                state_dict.pop(key)
        state_dict.pop(prefix + "edge_norm.weight", None)
        state_dict.pop(prefix + "proj.weight", None)
        state_dict.setdefault(prefix + "scale", self.scale.detach().clone())
        if self.normalize_residual:
            state_dict.setdefault(
                prefix + "angle_norm.weight",
                self.angle_norm.weight.detach().clone(),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        a: torch.Tensor,
        e: torch.Tensor,
        triplet_indices: torch.Tensor,
        support: AttentionSupport,
    ) -> torch.Tensor:
        if triplet_indices.numel() == 0:
            # Keep all parameters in the graph for empty-triplet DDP batches.
            zero = a.new_zeros(())
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0.0
            return a + zero

        if self.nonlinear_updates:
            pair_context = self.update(
                target=a,
                sources=e,
                pair_index=triplet_indices,
            )
        else:
            left, right = triplet_indices
            projected = self.proj(self.edge_norm(e))
            pair_context = (projected[left] + projected[right]) * (2.0**-0.5)
        residual_support = (
            support.positive.to(dtype=a.dtype)
            if self.support_mode == "mask"
            else support.value
        )
        updated = a + residual_support.unsqueeze(-1) * self.scale * pair_context
        return self.angle_norm(updated) if self.normalize_residual else updated


class InteractionLayer(nn.Module):
    """Node→edge feedback followed by line-graph and atom-graph attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        d_triplet: int | None = None,
        activation: str = "silu",
        use_swiglu: bool = False,
        dropout: float = 0.0,
        support_mode: str = "weighted",
        num_edge_basis: int | None = None,
        num_angle_basis: int | None = None,
        update_hidden_dim: int = 64,
        nonlinear_updates: bool = True,
        edge_update_initial_scale: float = 0.1,
        angle_update_initial_scale: float = 0.0,
        normalize_angle_update: bool = False,
    ):
        super().__init__()
        self.support_mode = support_mode
        d_triplet = d_triplet or d_model
        self.d_triplet = d_triplet

        self.node_to_edge_adapter = NodeToEdgeAdapter(
            d_model=d_model,
            support_mode=support_mode,
            initial_scale=edge_update_initial_scale,
            update_hidden_dim=update_hidden_dim,
            nonlinear_updates=nonlinear_updates,
        )
        self.angle_update = SymmetricAngleUpdate(
            d_model=d_model,
            d_triplet=d_triplet,
            support_mode=support_mode,
            initial_scale=angle_update_initial_scale,
            update_hidden_dim=update_hidden_dim,
            nonlinear_updates=nonlinear_updates,
            normalize_residual=normalize_angle_update,
        )

        # Convolve Angles → Edges (line graph)
        self.line_graph_conv = GraphTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_edge=d_triplet,
            dim_feedforward=dim_feedforward,
            activation=activation,
            use_swiglu=use_swiglu,
            dropout=dropout,
            support_mode=support_mode,
            d_rbf=num_angle_basis,
        )

        # Convolve Edges → Nodes (atom graph)
        self.atom_graph_conv = GraphTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_edge=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            use_swiglu=use_swiglu,
            dropout=dropout,
            support_mode=support_mode,
            d_rbf=num_edge_basis,
        )

    def forward(
        self,
        x,
        edge_index,
        e,
        edge_support: AttentionSupport,
        triplet_indices,
        a,
        triplet_support: AttentionSupport,
        edge_rbf,
        angle_rbf,
    ):
        e = self.node_to_edge_adapter(x, e, edge_index, edge_support)
        a = self.angle_update(a, e, triplet_indices, triplet_support)

        e = self.line_graph_conv(
            e,
            triplet_indices,
            a,
            triplet_support,
            angle_rbf,
        )
        x = self.atom_graph_conv(x, edge_index, e, edge_support, edge_rbf)

        return x, e, a


def support_weighted_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    support: torch.Tensor | None,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Softmax with continuous support entering numerator and denominator."""
    return _support_weighted_softmax_impl(scores, index, support, num_nodes=num_nodes)


def support_weighted_pool(
    feats: torch.Tensor,
    scores: torch.Tensor,
    index: torch.Tensor,
    size: int,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = support_weighted_softmax(scores, index, support, num_nodes=size)
    return global_add_pool(feats * weights, index, size=size)
