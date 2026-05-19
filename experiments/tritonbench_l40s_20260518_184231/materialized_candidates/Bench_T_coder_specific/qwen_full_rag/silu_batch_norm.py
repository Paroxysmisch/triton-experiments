import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def silu_swiglu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_batch_stride,
    a_feat_stride,
    c_batch_stride,
    c_feat_stride,
    N_CTX: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    offs_f = tl.arange(0, BLOCK_F)
    mask_f = offs_f < N_CTX
    a_ptrs = a_ptr + pid_b * a_batch_stride + offs_f[:, None] * a_feat_stride
    b_ptrs = b_ptr + pid_b * a_batch_stride + offs_f[:, None] * a_feat_stride
    c_ptrs = (
        c_ptr
        + pid_b * c_batch_stride
        + (offs_f[:, None] // 2) * c_feat_stride
        + ((offs_f % 2) * (N_CTX // 2))
    )

    a = tl.load(a_ptrs, mask=mask_f, other=0.0)
    b = tl.load(b_ptrs, mask=mask_f, other=0.0)

    swish = a * tl.sigmoid(a)
    c = swish * b

    tl.store(c_ptrs, c, mask=mask_f)

class SwiGLU_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, pre_act_cache=None, cache_key="default"):
        if not x.is_contiguous() or x.ndim != 2:
            raise ValueError("x must be contiguous 2D tensor")

        if pre_act_cache is None:
            pre_act_cache = torch.empty_like(x)
        elif (pre_act_cache is not None) and (
            (x.ndim != pre_act_cache.ndim) or (x.shape[-1] != pre_act_cache.shape[-1])
        ):
            raise ValueError(
                "pre_act_cache must be either None or a contiguous tensor with the same shape as x"
            )

        feat_size = x.shape[-1]
        batch_size = x.numel() // feat_size

        N_CTX = feat_size // 2
        BLOCK_N_CTX = max(triton.next_power_of_2(N_CTX), 16)

        if x.stride(-1) != 1:
            x = x.contiguous()

        if a is not None:
            if a.ndim != 2:
                raise ValueError("If provided, gate must be a 2D tensor")
            if (a.shape[-1] != feat_size) or (a.numel() != batch_size * feat_size):
                raise ValueError(
                    "gate must have the same number of elements as x, i.e., <batch_size, feat_size>"
                )
            if a.stride(-1) != 1:
                a = a.contiguous()

        if pre_act_cache.ndim != 2:
            raise ValueError("If provided, pre_act_cache must be a 2D tensor")
        if (pre_act_cache.shape[-1] != (feat_size // 2)) or (
            pre_act_cache.numel() != batch_size * (feat_size // 2)
        ):
            raise ValueError(
                "pre_act_cache must have the same number of elements as half the number of elements in x, i.e., <batch_size, feat_size // 2>"
            )
        if pre_act_cache.stride(-1) != 1:
            pre_act_cache = pre_act_cache.contiguous()

        grid = (batch_size,)
        kwargs = [
            x,
            a,
            pre_act_cache,
            x.batch_stride,
            x.feat_stride,
            pre_act_cache.batch_stride,
            pre_act_cache.feat_stride,
            N_CTX,
            BLOCK_N_CTX,
        ]

        silu_swiglu_kernel[grid](*kwargs)

        return pre_act_cache

def silu_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    # Function implementation
    pass
