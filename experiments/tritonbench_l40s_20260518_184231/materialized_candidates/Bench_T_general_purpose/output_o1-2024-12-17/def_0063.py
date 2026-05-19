import triton
import triton.language as tl
import torch
from typing import Union, Tuple, List

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Pointer arithmetic
    A_ptrs = A_ptr + (rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    B_ptrs = B_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(rm[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K), other=0.)
        b = tl.load(B_ptrs, mask=(rn[None, :] < N) & (k + tl.arange(0, BLOCK_K)[:, None] < K), other=0.)
        accum += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accum, mask=(rm[:, None] < M) & (rn[None, :] < N))

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    if isinstance(dims, int):
        a_sum_dims = list(range(a.ndim - dims, a.ndim))
        b_sum_dims = list(range(dims))
    else:
        a_sum_dims, b_sum_dims = dims

    # Reorder so that contraction dims are last for 'a' and first for 'b'
    a_other_dims = [i for i in range(a.ndim) if i not in a_sum_dims]
    b_other_dims = [i for i in range(b.ndim) if i not in b_sum_dims]
    a_perm = a_other_dims + a_sum_dims
    b_perm = b_sum_dims + b_other_dims
    a_trans = a.permute(*a_perm)
    b_trans = b.permute(*b_perm)

    # Flatten
    a_shape = a_trans.shape
    b_shape = b_trans.shape
    M = 1
    for x in a_shape[:-len(a_sum_dims)]:
        M *= x
    K = 1
    for x in a_shape[-len(a_sum_dims):]:
        K *= x
    N = 1
    for x in b_shape[len(b_sum_dims):]:
        N *= x
    a_reshaped = a_trans.reshape(M, K)
    b_reshaped = b_trans.reshape(K, N)

    # Prepare output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda META: ( (M + META['BLOCK_M'] - 1)//META['BLOCK_M'], 
                          (N + META['BLOCK_N'] - 1)//META['BLOCK_N'] )

    # Launch Triton kernel
    _matmul_kernel[grid](
        a_reshaped, b_reshaped, c,
        M, N, K,
        a_reshaped.stride(0), a_reshaped.stride(1),
        b_reshaped.stride(0), b_reshaped.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Reshape back
    out_shape = [a_shape[i] for i in range(len(a_shape) - len(a_sum_dims))] + [b_shape[i] for i in range(len(b_sum_dims), len(b_shape))]
    return c.reshape(*out_shape)
