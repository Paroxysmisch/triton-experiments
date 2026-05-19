import torch
import triton
import triton.language as tl
from flash_attn.ops.triton_k import einsum_m, einsum_m_core

@triton.jit
def rms_norm_kernel(
    X,  # shape: [B, S, H_0_padded]
    W,  # shape: [H_0_padded]
    Y,  # shape: [B, S, H_0_padded]
    H_0,
    B,
    S,
    bstride,
    wstride,
    ystride,
    eps,
    H_0_padded,
    bidx: tl.constexpr,
):
    # [overview]
    # - a single program processes a single example in parallel
    # - we process rows in a batch across multiple warps
    # - use einsum-m to perform multi-dimensional transposed matmul
    # - result is further broadcasted to all tokens in row

    pid = tl.program_id(0)
    padded_token_idx = tl.arange(0, H_0_padded)
    mask = padded_token_idx < H_0

    # [get example and position IDs, + broadcast IDs to tl.shape]
    eidx = tl.full([H_0_padded,], bidx, tl.int32)
    fidx = tl.full([H_0_padded,], 0, tl.int32)

    w = tl.load(W + padded_token_idx * wstride, mask=mask).to(tl.float32)

    # [broadcast stride and idx over large token dim, for einsum]
    cstride = tl.full([H_0_padded,], 1, tl.int32)
    cvidx = tl.arange(0, H_0_padded)
    cwidx = tl.full([H_0_padded,], 0, tl.int32)

    x = tl.load(
        X + padded_token_idx * xstride + pid * bstride, mask=mask
    ).to(tl.float32)  # [H_0_padded]

    # [compute variance, unexpected dims are summed over]
    x = x / tl.sqrt(tl.sum(x * x) / H_0 + eps)
    x = x * w

    # [store]
    tl.store(
        Y + padded_token_idx * ystride + pid * bstride, x, mask=mask
    )  # [H_0_padded]

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-6):
        B = x.shape[0]
        S = x.shape[1]
        H_0 = normalized_shape[-1]
        H_0_padded = einsum_m_core(H_0, 16)
        y = torch.empty_like(x)

        def grid(meta):
            return (triton.cdiv(H_0_padded, meta["BLOCK"]), B * S)

        rms_norm_kernel[grid](x, weight, y, H_0, B, S, bstride=x.stride(0), wstride=weight.stride(0), ystride=y.stride(0), eps=eps, H_0_padded=H_0_padded)
        return y

def rms_norm(x, normalized_shape, weight, eps=1e-6):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
