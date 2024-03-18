"""\
the code below is a reimplentation of https://github.com/xai-org/grok-1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import NamedTuple
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

    def __init__(self, dim: int, seq_len: int = 2048, theta: float = 10000.0) -> None:
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


class MultiLayerPerceptron(nn.Module):
    dim: int
    hidden_dim: int

    lin_in: nn.Linear
    lin_out: nn.Linear
    act: nn.GELU

    def __init__(
        self, dim: int, hidden_dim: int | None, expansion_ratio: float = 1.0
    ) -> None:
        """\
        if hidden_dim is not specified, it will be set to dim * expansion_ratio else expansion_ratio will be ignored
        """
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * expansion_ratio)

        self.lin_in = nn.Linear(self.dim, self.hidden_dim)
        self.lin_out = nn.Linear(self.hidden_dim, self.dim)
        self.act = nn.GELU()  # TODO support other activations

    def forward(
        self, x: Annotated[torch.Tensor, ..., "D"]
    ) -> Annotated[torch.Tensor, ..., "D"]:
        return self.lin_out(self.act(self.lin_in(x)))


class AttentionLinearBias(nn.Module):
    """\
    implementation of: https://arxiv.org/abs/2108.12409
    """

    _cached_seq_len: int
    _bias_buffer: Annotated[torch.Tensor, "T", "T"]
    _factor: float

    def __init__(self, seq_len: int, factor: float = 1 / (2**8)) -> None:
        super().__init__()
        self._factor = factor
        self._cached_seq_len = seq_len
        self.register_buffer("_bias_buffer", self._get_bias(seq_len))

    @staticmethod
    def _get_bias(
        l: int, device: torch.device | None = None
    ) -> Annotated[torch.Tensor, "T", "T"]:
        a = torch.arange(0, l, device=device or "cpu").reshape(-1, 1)
        return -torch.relu(a - a.T)

    def forward(
        self, seq_len: int, n_heads: int, device: torch.device
    ) -> Annotated[torch.Tensor, "T", "T", "H"]:
        if seq_len > self._cached_seq_len:
            next_power_of_two = 2 ** math.ceil(math.log2(seq_len))
            self._cached_seq_len = next_power_of_two
            self.register_buffer(
                "_bias_buffer", self._get_bias(next_power_of_two, device)
            )
        bias = self._bias_buffer[:seq_len, :seq_len].reshape(-1, -1, 1)
        head_factor = self._factor ** (1 / n_heads)
        k = torch.pow(head_factor, torch.arange(0, n_heads, device=device)).reshape(
            1, 1, -1
        )
        return bias * k


class MultiHeadAttention(nn.Module):
    """\
    implementation of: https://arxiv.org/abs/1706.03762

    some implementation details were taken from https://github.com/lucidrains/x-transformers
    which is licended under the...

    MIT License

    Copyright (c) 2020 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    """

    dim: int
    value_dim: int
    out_dim: int

    num_heads: int
    k_dim_head: int
    v_dim_head: int

    causal: bool
    use_flash: bool

    rotary_pos_emb: RoPE | None
    attention_linear_bias: AttentionLinearBias | None

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        use_flash: bool = False,
        num_heads: int = 8,
        k_dim_head: int = 64,
        v_dim_head: int = 64,
        value_dim: int | None = None,
        out_dim: int | None = None,
        rotary_pos_emb: RoPE | None = None,
        attention_linear_bias: AttentionLinearBias | None = None,
    ) -> None:
        super().__init__()

        if value_dim is None:
            value_dim = dim
        if out_dim is None:
            out_dim = value_dim

        self.dim = dim
        self.value_dim = value_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        self.k_dim_head = k_dim_head
        self.v_dim_head = v_dim_head

        self.causal = causal
        self.use_flash = use_flash  # TODO implement flash

        # positional embedding
        assert (
            rotary_pos_emb is None or attention_linear_bias is None
        ), "can't use RoPE and ALiBi at the same time"
        self.rotary_pos_emb = rotary_pos_emb
        self.attention_linear_bias = attention_linear_bias

        v_dim = self.v_dim_head * self.num_heads
        q_dim = k_dim = self.k_dim_head * self.num_heads

        self.w_q = nn.Linear(self.dim, q_dim, bias=False)
        self.w_k = nn.Linear(self.dim, k_dim, bias=False)
        self.w_v = nn.Linear(self.value_dim, v_dim, bias=False)
        self.w_o = nn.Linear(v_dim, self.out_dim, bias=False)

    def forward(
        self,
        q: Annotated[torch.Tensor, ..., "T", "K"],
        k: Annotated[torch.Tensor, ..., "T", "K"],
        v: Annotated[torch.Tensor, ..., "T", "V"],
        mask: Annotated[torch.Tensor, "T", "T"] | None = None,
    ) -> Annotated[torch.Tensor, ..., "T", "O"]:
        assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
        assert q.shape[-1] == k.shape[-1] == self.dim
        assert v.shape[-1] == self.value_dim
        if mask is not None:
            assert mask.shape == (q.shape[-2], k.shape[-2])
        B = q.shape[:-2]  # batch shape

        q_i = rearrange(self.w_q(q), "... T (H k) -> ... H T k", H=self.num_heads)
        k_i = rearrange(self.w_k(k), "... T (H k) -> ... H T k", H=self.num_heads)
        v_i = rearrange(self.w_v(v), "... T (H v) -> ... H T v", H=self.num_heads)

        if self.rotary_pos_emb is not None:
            rope_dim = self.rotary_pos_emb.dim

            def _apply(x: torch.Tensor) -> torch.Tensor:
                if TYPE_CHECKING:
                    assert self.rotary_pos_emb is not None
                return torch.cat(
                    (self.rotary_pos_emb(x[..., :rope_dim]), x[..., rope_dim:]), dim=-1
                )

            q_i = _apply(q_i)
            k_i = _apply(k_i)
            v_i = _apply(v_i)

        # use scaled dot product similarity
        s_qk = einsum(q_i, k_i, "... H i k, ... H j k -> ... H i j")

        s_qk = s_qk / (q_i.shape[-1] ** 0.5)

        # apply mask
        if mask is not None:
            mask = mask.view(*B, *mask.shape)
            mask_value = -torch.finfo(s_qk.dtype).max
            s_qk = s_qk.masked_fill(~mask, mask_value)

        # softmax
        attn: Annotated[torch.Tensor, ..., "H", "T", "T"] = F.softmax(s_qk, dim=-1)

        vals = einsum(attn, v_i, "... H T i, ... H i v -> ... H T v")
        out = self.w_o(rearrange(vals, "... H T v -> ... T (H v)"))
        return out


class MultiHeadAttentionBlock(nn.Module):
    dim: int

    attn: MultiHeadAttentionBlock
    mlp: MultiLayerPerceptron

    pre_norm: RMSNorm
    post_norm: RMSNorm

    attn_dropout: nn.Dropout | None = None
    mem_dropout: nn.Dropout | None = None
    dropout: nn.Dropout | None = None

    def __init__(
        self,
        dim: int,
        dropout: float | None = None,
        attn_dropout: float | None = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.attn = MultiHeadAttentionBlock(self.dim)
        self.mlp = MultiLayerPerceptron(self.dim, hidden_dim=4 * self.dim)

        self.pre_norm = RMSNorm(self.dim)
        self.post_norm = RMSNorm(self.dim)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        if attn_dropout is not None:
            self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self, x: Annotated[torch.Tensor, ..., "B", "T", "D"]
    ) -> Annotated[torch.Tensor, ..., "T", "D"]:
        k = self.pre_norm(x)
        attn = self.attn(k, k, k)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        r = k + attn
        m = self.post_norm(r)
        m = self.mlp(m)

        if self.dropout is not None:
            m = self.dropout(m)

        return r + m


class Router(nn.Module):
    num_experts: int
    num_selected_experts: int
    sequence_length: int

    def __init__(
        self, num_experts: int, num_selected_experts: int, sequence_length: int
    ) -> None:
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.sequence_length = sequence_length

        self.lin = nn.Linear(sequence_length, num_experts, bias=False)

    def forward(
        self, x: Annotated[torch.Tensor, "B", "T", "D"]
    ) -> Annotated[torch.Tensor, "B", "M", "D"]:
        logits = self.lin(x)
        return F.softmax(logits, dim=-1)


class MixtureOfExpertsLayer(nn.Module):
    num_experts: int
    router: Router

    def __init__(self, num_experts: int, router: Router) -> None:
        self.num_experts = num_experts
        self.router = router

    def forward(
        self, x: Annotated[torch.Tensor, ..., "B", "T", "D"]
    ) -> Annotated[torch.Tensor, ..., "B", "T", "D"]:
        ...


class DenseBlock(nn.Module):
    ...


class DecoderLayer(nn.Module):
    ...


@dataclass
class TransformerConfig:
    ...


@dataclass
class LanguageModelConfig:
    ...


class Transformer(nn.Module):
    ...


class LanguageModel(nn.Module):
    ...
