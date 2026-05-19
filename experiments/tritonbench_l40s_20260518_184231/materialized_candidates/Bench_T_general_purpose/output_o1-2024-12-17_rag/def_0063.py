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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # k-range used in reduction
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers
    a_ptrs = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
    
    c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Reduction loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        c_acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write output
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c_acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Handle dims argument: convert any form into two lists of axes
    def _parse_dims(a_shape, b_shape, dims):
        if isinstance(dims, int):
            # Last dims of a, first dims of b
            return list(range(len(a_shape) - dims, len(a_shape))), list(range(dims))
        if isinstance(dims, (tuple, list)):
            if isinstance(dims[0], int):
                return [dims[i] for i in range(len(dims))], [dims[i] for i in range(len(dims))]
            # assume tuple/list of lists
            return dims[0], dims[1]
        raise RuntimeError("Invalid dims argument")

    a_shape = list(a.shape)
    b_shape = list(b.shape)

    # Get contraction dims
    a_dims, b_dims = _parse_dims(a_shape, b_shape, dims)
    # Make them positive and sorted for convenience
    a_dims = [x if x >= 0 else x + len(a_shape) for x in a_dims]
    b_dims = [x if x >= 0 else x + len(b_shape) for x in b_dims]
    a_dims.sort()
    b_dims.sort()

    # Check match or broadcast in contracted dimensions
    for ad, bd in zip(a_dims, b_dims):
        if a_shape[ad] != b_shape[bd] and not (a_shape[ad] == 1 or b_shape[bd] == 1):
            raise RuntimeError("Contracted shapes must match or be broadcastable")

    # Compute the shape of the result
    a_outshape = [a_shape[i] for i in range(len(a_shape)) if i not in a_dims]
    b_outshape = [b_shape[i] for i in range(len(b_shape)) if i not in b_dims]

    # Compute the broadcasted "K" dimension from a/b contraction
    k_shape = []
    for ad, bd in zip(a_dims, b_dims):
        k_shape.append(max(a_shape[ad], b_shape[bd]))
    # Flatten that "K" dimension
    K = 1
    for ks in k_shape:
        K *= ks

    # Flatten "M"; from a_outshape
    M = 1
    for s in a_outshape:
        M *= s
    # Flatten "N"; from b_outshape
    N = 1
    for s in b_outshape:
        N *= s

    # Reshape a and b into (M,K) and (K,N)
    # Expand dims for broadcast if needed
    a_expand_shape = []
    b_expand_shape = []
    idx_a = 0
    idx_b = 0
    # Insert "1" in place of contracted dims for matching
    for i in range(len(a_shape)):
        if i in a_dims:
            a_expand_shape.append(k_shape[idx_a])  # broadcast dimension
            idx_a += 1
        else:
            a_expand_shape.append(a_shape[i])
    for i in range(len(b_shape)):
        if i in b_dims:
            b_expand_shape.append(k_shape[idx_b])  # broadcast dimension
            idx_b += 1
        else:
            b_expand_shape.append(b_shape[i])

    # Expand if needed
    a_expanded = a.expand(a_expand_shape)
    b_expanded = b.expand(b_expand_shape)

    a_2d = a_expanded.reshape(M, K)
    b_2d = b_expanded.reshape(K, N)

    # Prepare output tensor
    c = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _tensordot_kernel[grid](
        a_2d, b_2d, c,
        M, N, K,
        a_2d.stride(0), a_2d.stride(1),
        b_2d.stride(0), b_2d.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )

    # Reshape output to final
    out = c.reshape(*a_outshape, *b_outshape)
    return out
