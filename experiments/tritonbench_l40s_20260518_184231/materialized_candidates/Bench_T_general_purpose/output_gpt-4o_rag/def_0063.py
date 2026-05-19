import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List

@triton.jit
def _tensordot_kernel(
    A, B, C, 
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, c)

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    if isinstance(dims, int):
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))
    elif isinstance(dims, tuple) or isinstance(dims, list):
        dims_a, dims_b = dims

    # Permute dimensions to bring contraction dimensions to the end for `a` and to the beginning for `b`
    a_perm = [i for i in range(a.ndim) if i not in dims_a] + dims_a
    b_perm = dims_b + [i for i in range(b.ndim) if i not in dims_b]

    a = a.permute(a_perm)
    b = b.permute(b_perm)

    # Calculate the shape of the result tensor
    shape_a = a.shape
    shape_b = b.shape
    M = int(torch.prod(torch.tensor(shape_a[:-len(dims_a)])))
    N = int(torch.prod(torch.tensor(shape_b[len(dims_b):])))
    K = int(torch.prod(torch.tensor(shape_a[-len(dims_a):])))

    # Reshape tensors to 2D for the matrix multiplication
    a_reshaped = a.reshape(M, K)
    b_reshaped = b.reshape(K, N)

    # Prepare output tensor
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Calculate strides
    stride_am, stride_ak = a_reshaped.stride()
    stride_bk, stride_bn = b_reshaped.stride()
    stride_cm, stride_cn = c.stride()

    # Launch Triton kernel
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tensordot_kernel[grid](
        a_reshaped, b_reshaped, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Reshape the result back to the correct dimensions
    result_shape = a.shape[:-len(dims_a)] + b.shape[len(dims_b):]
    return c.reshape(result_shape)
