import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK: tl.constexpr,
):
    # Kernel code omitted for brevity.
    pass

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, sm_scale, Out,
    DO, DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # Kernel code omitted for brevity.
    pass

@triton.jit
def _bwd_inter_kernel(
    Q, K, V, sm_scale, Out,
    DO, DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    BLOCK: tl.constexpr,
):
    # Kernel code omitted for brevity.
    pass

class LightningAttention2NoDecay(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK = 64
        # Kernel launch omitted for brevity.
        pass
        ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        BLOCK = 64
        # Kernel launch omitted for brevity.
        pass
        return dq, dk, dv

def lightning_attn2_no_decay(q, k, v):
    return LightningAttention2NoDecay.apply(q, k, v)
