import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x shape: bsz, seqlen, self.n_local_heads, self.head_hidden_dim / 2
    # the last shape is "self.hidden_dim / 2" because we convert to complex
    assert x.ndim == 4
    assert freqs_cis.shape == (
        x.shape[0],
        x.shape[1],
        x.shape[-1],
    ), f"x shape: {x.shape}, freqs_cis shape: {freqs_cis.shape}"

    # reshape freq cis to match and apply pointwise multiply
    # new shape: bsz, seq_len, 1, self.head_hidden_dim / 2
    shape = [x.shape[0], x.shape[1], 1, x.shape[-1]]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        flash_attention: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = flash_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        seq_len,
        cu_seqlens,    # cu_seqlens = torch.cumsum(self.seq_len, dim=0)，见 ./kimia_infer/models/detokenizer/flow_matching/ode_wrapper.py， 即 [1,2,3,4,5] => [1, 3, 6, 10, 15]
                       #   cu_seqlens 和 cu_seqlens_k 是为了支持 变长序列批处理（variable-length batching），也就是：把一个 batch 中 多个不同长度的序列 拼接成一个长张量，并用 cu_seqlens 来记录每条序列的起始位置。
                       #   这是 FlashAttention 中使用的一种高效 batch 表示方式。
        max_seqlen,
        cu_seqlens_k,  # 用于flash-attention
        max_seqlen_k,
        rotary_pos_emb=None,
        incremental_state=None,
        nopadding=True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        # 因为 x.shape = (batch_size, seq_len, fea_dim), 所以 attention 内部是像 LLM 的 tokens 之间作 attention 一样做的

        if self.fused_attn: ##  = flash_attention
            if nopadding:
                qkv = self.qkv(x)
                qkv = qkv.view(B * N, self.num_heads * 3, self.head_dim)
                q, k, v = qkv.split([self.num_heads] * 3, dim=1)
                q, k = self.q_norm(q), self.k_norm(k)

                q = q.view(B, N, self.num_heads, self.head_dim)
                k = k.view(B, N, self.num_heads, self.head_dim)
                v = v.view(B, N, self.num_heads, self.head_dim)

                if rotary_pos_emb is not None:
                    q, k = apply_rotary_emb(q, k, rotary_pos_emb)

                if incremental_state is not None:
                    if "prev_k" in incremental_state:
                        prev_k = incremental_state["prev_k"]
                        k = torch.cat([prev_k, k], dim=1)

                    if "cur_k" not in incremental_state:
                        incremental_state["cur_k"] = {}
                    incremental_state["cur_k"] = k

                    if "prev_v" in incremental_state:
                        prev_v = incremental_state["prev_v"]
                        v = torch.cat([prev_v, v], dim=1)

                    if "cur_v" not in incremental_state:
                        incremental_state["cur_v"] = {}
                    incremental_state["cur_v"] = v

                q = q.view(B * N, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_heads, self.head_dim)
                v = v.view(-1, self.num_heads, self.head_dim)

                x = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:

                if incremental_state is not None:
                    raise NotImplementedError(
                        "It is designed for batching inference. AR-chunk is not supported currently."
                    )

                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
                if self.qk_norm:
                    q, k, v = qkv.unbind(2)
                    q, k = self.q_norm(q), self.k_norm(k)
                    # re-bind
                    qkv = torch.stack((q, k, v), dim=2)

                # pack qkv with seq_len
                qkv_collect = []
                for i in range(qkv.shape[0]):
                    qkv_collect.append(qkv[i, : seq_len[i], :, :, :])

                qkv = torch.cat(qkv_collect, dim=0)

                x = flash_attn_varlen_qkvpacked_func(
                    qkv=qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )

                # unpack and pad 0
                x_collect = []
                for i in range(B):
                    x_collect.append(x[cu_seqlens[i] : cu_seqlens[i + 1], :, :])
                x = torch.nn.utils.rnn.pad_sequence(
                    x_collect, batch_first=True, padding_value=0
                )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        ffn_type="conv1d_conv1d", # 但是并不用它，而是用的 valilla_mlp
        ffn_gated_glu=True,
        ffn_act_layer="gelu",
        ffn_conv_kernel_size=5,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if ffn_type == "vanilla_mlp":
            from timm.models.vision_transformer import Mlp

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        else:
            raise NotImplementedError(f"FFN type {ffn_type} is not implemented")

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x,
        c,
        seq_len,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        mask,
        rotary_pos_emb=None,
        incremental_state=None,
        nopadding=True,
    ):
        # x.shape = torch.Size([1, 120, 2304]) c.shape = torch.Size([1, 120, 2304])
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=2)    # Silu + linear， 条件c只用于这里
        )

        ############  ATTN ##############
        
        x_ = modulate(self.norm1(x), shift_msa, scale_msa)  # => x * (1 + scale) + shift， scale shift来自c

        if incremental_state is not None:
            if "attn_kvcache" not in incremental_state:
                incremental_state["attn_kvcache"] = {}
            inc_attn = incremental_state["attn_kvcache"]
        else:
            inc_attn = None

        x_ = self.attn(x_,   # x.shape = (B, N, C), 所以 attn 是N的不同元素之间做的，也就是类似 LLM 的不同token之间那样。
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            max_seqlen=cu_maxlen,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=cu_maxlen_k,
            rotary_pos_emb=rotary_pos_emb,
            incremental_state=inc_attn,
            nopadding=nopadding,
        )

        if not nopadding:
            x_ = x_ * mask[:, :, None]

        x = x + gate_msa * x_ # x 是原始input x， gate_msa 来自c

        ############  MLP ##############
        
        x_ = modulate(self.norm2(x), shift_mlp, scale_mlp) # shift_* 来自c

        x_ = self.mlp(x_)

        if not nopadding:
            x_ = x_ * mask[:, :, None]

        x = x + gate_mlp * x_ # gate_mlp 来自 c
        return x
