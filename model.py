"""\
the code below is a reimplentation of:
    https://github.com/xai-org/grok-1
lincesed under:
    Apache License 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Annotated
from einops import rearrange, einsum
import math
from typing import TYPE_CHECKING


class RMSNorm(nn.Module):
    """\
    implementation of: https://arxiv.org/abs/1910.07467
    """

    eps: float
    dim: int
    create_scale: bool

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        create_scale: bool = True,
    ) -> None:
        super().__init__()

        self.var_eps = eps
        self.dim = dim

        self.create_scale = create_scale
        self.weight = nn.Parameter(torch.ones(dim)) if create_scale else None

    def forward(self, x: Annotated[torch.Tensor, ...]) -> Annotated[torch.Tensor, ...]:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps)

        if self.create_scale:
            assert self.weight is not None
            if self.weight.dtype != x.dtype:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


class RoPE(nn.Module):
    """\
    implementation of: https://arxiv.org/abs/2104.09864
    """

    theta: float
    dim: int

    _seq_len_cached: int
    _freqs_cis: Annotated[torch.Tensor, torch.complex64, "T", "D"]
    _scale: Annotated[torch.Tensor, "D"]

    def __init__(self, dim: int, seq_len: int = 8192, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

        assert seq_len >= 1
        freqs_cis = self._get_freqs_cis(seq_len)
        self.register_buffer("_freqs_cis", freqs_cis)

    def _get_freqs_cis(
        self, seq_len: int, device: torch.device | None = None
    ) -> Annotated[torch.Tensor, torch.complex64, "T", "D"]:
        self._seq_len_cached = seq_len
        half = self.dim // 2  # only apply to half of the dimensions, see the paper
        freqs = self.theta ** -(
            torch.arange(0, half, device=device or "cpu").float() / half
        )
        seq = torch.arange(seq_len, device=freqs.device)
        freqs = einsum(seq, freqs, "T, D -> T D")
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def get_freqs_cis(
        self, seq_len: int, device: torch.device
    ) -> Annotated[torch.Tensor, torch.complex64, "T", "D/2"]:
        if seq_len > self._seq_len_cached:
            next_power_of_two = 2 ** math.ceil(math.log2(seq_len))
            freqs_cis = self._get_freqs_cis(next_power_of_two, device=device)
            self.register_buffer("_freqs_cis", freqs_cis)
        return self._freqs_cis[-seq_len:, :]

    @staticmethod
    def rotate_half(
        x: Annotated[torch.Tensor, ..., "D"]
    ) -> Annotated[torch.Tensor, ..., "D"]:
        x = rearrange(x, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: Annotated[torch.Tensor, ..., "T", "D"],
    ) -> Annotated[torch.Tensor, ..., "T", "D"]:
        """applies rotary embeddings to x"""
        freqs_cis = self.get_freqs_cis(x.shape[-2], device=x.device)
        assert x.shape[-1] == freqs_cis.shape[-1]

        freqs_cos = torch.view_as_real(freqs_cis)
        freqs_sin = torch.view_as_complex(freqs_cis)
        return (x * freqs_cos) + (self.rotate_half(x) * freqs_sin)


class DenseMultiLayerPerceptron(nn.Module):
    dim: int
    dim_inner: int

    def __init__(self, dim: int, dim_inner: int) -> None:
        super().__init__()

        self.dim = dim
        self.dim_inner = dim_inner

        self.lin_in = nn.Linear(dim, dim_inner)
        self.lin_out = nn.Linear(dim_inner, dim)
        self.lin_scale = nn.Linear(dim, dim_inner)

        self.act = nn.GELU()

    def forward(
        self, x: Annotated[torch.Tensor, ..., "D"]
    ) -> Annotated[torch.Tensor, ..., "D"]:
        inner = self.act(self.lin_in(x)) * self.lin_scale(x)
        return self.lin_out(inner)


class MultiHeadAttentionBlock(nn.Module):
    """\
    implementation of: https://arxiv.org/abs/1706.03762
    some implementation details were taken from:
        https://github.com/lucidrains/x-transformers
    which is licensed under the MIT License
    """

    dim: int
    k_dim_head: int
    v_dim_head: int

    num_heads: int
    num_q_heads: int

    causal: bool
    use_flash: bool

    rotary_pos_emb: RoPE

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        use_flash: bool = False,
        num_heads: int = 8,
        num_q_heads: int | None = None,
        k_dim_head: int = 128,
        v_dim_head: int = 128,
    ) -> None:
        super().__init__()

        self.dim = dim

        self.num_heads = num_heads
        self.num_q_heads = num_q_heads or num_heads
        assert (
            self.num_q_heads % self.num_heads == 0
            and self.num_q_heads >= self.num_heads
        ), "num_q_heads must be a proper multiple of num_heads"

        self.k_dim_head = k_dim_head
        self.v_dim_head = v_dim_head

        self.causal = causal
        self.use_flash = use_flash  # TODO implement flash

        # positional embedding
        self.rotary_pos_emb = RoPE(dim)

        v_dim = self.v_dim_head * self.num_heads
        k_dim = self.k_dim_head * self.num_heads
        q_dim = self.k_dim_head * self.num_q_heads

        self.w_q = nn.Linear(self.dim, q_dim, bias=False)
        self.w_k = nn.Linear(self.dim, k_dim, bias=False)
        self.w_v = nn.Linear(self.dim, v_dim, bias=False)
        self.w_o = nn.Linear(q_dim, self.dim, bias=False)

    def forward(
        self,
        q: Annotated[torch.Tensor, ..., "T", "K"],
        k: Annotated[torch.Tensor, ..., "T", "K"],
        v: Annotated[torch.Tensor, ..., "T", "V"],
        mask: Annotated[torch.Tensor, ..., "T", "T"] | None = None,
    ) -> Annotated[torch.Tensor, ..., "T", "O"]:
        assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
        assert q.shape[-1] == k.shape[-1] == self.dim
        assert v.shape[-1] == self.dim
        if mask is not None:
            assert mask.shape == (q.shape[-2], k.shape[-2])
        B = q.shape[:-2]  # batch shape

        q_i = rearrange(self.w_q(q), "... T (HQ k) -> ... HQ T k", HQ=self.num_q_heads)
        k_i = rearrange(self.w_k(k), "... T (HK k) -> ... HK T k", HK=self.num_heads)
        v_i = rearrange(self.w_v(v), "... T (HK v) -> ... HK T v", HK=self.num_heads)

        q_i = rearrange(q_i, "... HQ T k -> ... HK h T k", HK=self.num_heads)

        # apply rope
        def _apply_rope(
            x: Annotated[torch.Tensor, ...]
        ) -> Annotated[torch.Tensor, ...]:
            rope_dim = self.rotary_pos_emb.dim
            return torch.cat(
                (self.rotary_pos_emb(x[..., :rope_dim]), x[..., rope_dim:]), dim=-1
            )

        q_i = _apply_rope(q_i)
        k_i = _apply_rope(k_i)
        v_i = _apply_rope(v_i)

        # use scaled dot product similarity
        s_qk = einsum(q_i, k_i, "... HK h i k, ... HK j k -> ... HK h i j")
        s_qk = s_qk / (q_i.shape[-1] ** 0.5)

        # apply mask
        if mask is not None:
            mask = mask.view(*B, 1, *mask.shape)
            mask_value = -torch.finfo(s_qk.dtype).max
            s_qk = s_qk.masked_fill(~mask, mask_value)

        # softmax
        attn: Annotated[torch.Tensor, ..., "H", "T", "T"] = F.softmax(s_qk, dim=-1)

        vals = einsum(attn, v_i, "... HK h T i, ... HK i v -> ... HK h T v")
        vals = rearrange(vals, "... HK h T v -> ... T (HK h v)")
        out = self.w_o(vals)
        return out


class Router(nn.Module):
    dim: int
    num_experts: int

    def __init__(self, num_experts: int, dim: int) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.dim = dim
        self.lin = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self, states: Annotated[torch.Tensor, "(B T)", "D"]
    ) -> Annotated[torch.Tensor, "(B T)", "H"]:
        router_logits = self.gate(states)
        router_weights = F.softmax(router_logits, dim=-1)
        return router_weights


class MoEBlock(nn.Module):
    """\
    some code from this class is a rewrite of huggingface's implementation 
    of mixtral: huggingface/transformers/models/mixtral/modeling_mixtral.py
    ...which is licensed under the Apache License 2.0
    """

    dim: int
    dim_inner: int
    num_experts: int
    num_selected_experts: int

    router: Router
    experts: nn.ModuleList

    def __init__(
        self, dim: int, dim_inner: int, num_experts: int, num_selected_experts: int
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dim_inner = dim_inner

        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        self.router = Router(self.num_experts, self.dim)
        self.experts = nn.ModuleList(
            [
                DenseMultiLayerPerceptron(self.dim, self.dim_inner)
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self, states: Annotated[torch.Tensor, "B", "T", "D"]
    ) -> Annotated[torch.Tensor, "B", "T", "D"]:
        B, T, _ = states.shape
        states = rearrange(states, "B T D -> (B T) D")

        router_weights = self.router(states)

        # truncate top k and normalize
        router_weights, selected_experts = torch.topk(
            router_weights, self.num_selected_experts, dim=-1
        )
        router_weights /= router_weights.sum(dim=-1, keepdim=True)

        expert_mask = F.one_hot(selected_experts, self.num_experts)
        expert_mask = rearrange(expert_mask, "(B T) K H -> H K (B T)")

        ret = torch.zeros_like(states)

        for expert_idx in range(self.num_experts):
            mask: Annotated[torch.Tensor, "K", "(B T)"] = expert_mask[expert_idx]
            k_idx, batch_idx = torch.where(mask)

            if batch_idx.shape[0] == 0:
                continue  # if this expert is not used by any token across the batch

            batch_idx_lst = batch_idx.tolist()
            k_idx_lst = k_idx.tolist()

            tiled_states = rearrange(
                states[None, batch_idx_lst], "H (B T) D -> H (B T) D"
            )
            expert_states = self.experts[expert_idx](tiled_states)
            expert_states *= router_weights[batch_idx_lst, k_idx_lst, None]

            ret[batch_idx_lst, k_idx_lst, :] = expert_states  # does this work?

        return rearrange(ret, "(B T) D -> B T D", B=B, T=T)


class GrokLayer(nn.Module):
    dim: int
    dim_inner: int

    num_experts: int
    num_selected_experts: int

    attn: MultiHeadAttentionBlock
    block: DenseMultiLayerPerceptron | MoEBlock

    attn_pre_norm: RMSNorm
    moe_pre_norm: RMSNorm
    attn_post_norm: RMSNorm
    moe_post_norm: RMSNorm

    def __init__(
        self,
        dim: int,
        dim_inner: int,
        num_heads: int = 8,
        num_q_heads: int = 64,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        k_dim_head: int = 128,
        v_dim_head: int = 128,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dim_inner = dim_inner
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        self.attn = MultiHeadAttentionBlock(
            self.dim,
            num_heads=num_heads,
            num_q_heads=num_q_heads,
            k_dim_head=k_dim_head,
            v_dim_head=v_dim_head,
        )

        assert self.num_experts > self.num_selected_experts

        if self.num_experts <= 1:
            self.block = DenseMultiLayerPerceptron(self.dim, self.dim_inner)
        else:
            self.block = MoEBlock(
                dim=self.dim,
                dim_inner=self.dim_inner,
                num_experts=self.num_experts,
                num_selected_experts=self.num_selected_experts,
            )

        self.moe_pre_norm = RMSNorm(self.dim)
        self.attn_pre_norm = RMSNorm(self.dim)
        self.attn_post_norm = RMSNorm(self.dim)
        self.moe_post_norm = RMSNorm(self.dim)

    def forward(
        self,
        embeddings: Annotated[torch.Tensor, ..., "B", "T", "D"],
        mask: Annotated[torch.Tensor, "B", "T", "T"] | None = None,
    ) -> Annotated[torch.Tensor, ..., "T", "D"]:
        # attention
        key = self.attn_pre_norm(embeddings)
        attn = self.attn(key, key, key, mask=mask)
        embeddings += self.attn_post_norm(attn)

        # mixture of experts
        moe = self.moe_pre_norm(embeddings)
        moe = self.block(moe)
        embeddings += self.moe_post_norm(moe)

        return embeddings


class Grok(nn.Module):
    dim: int
    dim_inner: int

    num_layers: int

    num_experts: int
    num_selected_experts: int

    num_heads: int
    num_q_heads: int
    k_dim_head: int
    v_dim_head: int

    def __init__(
        self,
        dim: int = 6144,
        widening_factor: int = 8,
        num_layers: int = 64,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        num_heads: int = 8,
        num_q_heads: int = 48,
        k_dim_head: int = 128,
        v_dim_head: int = 128,
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = dim * widening_factor

        self.num_layers = num_layers

        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        self.num_heads = num_heads
        self.num_q_heads = num_q_heads
        self.k_dim_head = k_dim_head
        self.v_dim_head = v_dim_head

        # I don't like big multi-line list comprehensions :(
        _layers = []
        for _ in range(self.num_layers):
            layer = GrokLayer(
                dim=self.dim,
                dim_inner=self.dim_inner,
                num_heads=self.num_heads,
                num_q_heads=self.num_q_heads,
                num_experts=self.num_experts,
                num_selected_experts=self.num_selected_experts,
                k_dim_head=self.k_dim_head,
                v_dim_head=self.v_dim_head,
            )
            _layers.append(layer)

        self.layers = nn.ModuleList(_layers)

    def forward(
        self,
        embeddings: Annotated[torch.Tensor, "B", "T", "D"],
        token_mask: Annotated[torch.Tensor, "B", "T"] | None = None,
    ) -> Annotated[torch.Tensor, "B", "T", "D"]:
        B, T, D = embeddings.shape

        assert D == self.dim
        assert token_mask is None or token_mask.shape == (B, T)

        if token_mask is None:
            token_mask = torch.ones(B, T, dtype=torch.bool, device=embeddings.device)

        # causal masking for autoregressive language modeling
        mask: Annotated[torch.Tensor, "B", "T", "T"] = token_mask[:, None, :]
        causal_mask = torch.triu(torch.ones((1, T, T), device=mask.device))
        mask = mask * causal_mask

        for layer in self.layers:
            embeddings = layer(embeddings, mask=mask)

        return embeddings
