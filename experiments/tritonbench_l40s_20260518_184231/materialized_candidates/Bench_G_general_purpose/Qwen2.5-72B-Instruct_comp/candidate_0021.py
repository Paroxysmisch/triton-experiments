import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A, B, O,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr
):
    # Compute the tile indices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_k = tl.cdiv(K, TILE_K)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = GROUP_M * group_id
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the tile offsets
    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)

    # Initialize the output tile
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, K, TILE_K):
        # Load the tiles of A and B
        offs_ak = (k + offs_k) % K
        A_tile = tl.load(A + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak), mask=offs_am[:, None] < M, other=0.0)
        B_tile = tl.load(B + (offs_ak[:, None] * stride_bk + offs_bn[None, :] * stride_bn), mask=offs_bn[None, :] < N, other=0.0)

        # Perform the matrix multiplication
        acc += tl.dot(A_tile, B_tile)

    # Store the result in the output tensor
    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(O + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on), acc, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 16, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=16, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm(A, B, O, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_om, stride_on):
    bmm_kernel[A, B, O, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_om, stride_on]

def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Get the dimensions
    batch, M, K = A.shape
    batch, K, N = B.shape

    # Initialize the output tensor
    O = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)

    # Determine the grid dimensions
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 16), batch)

    # Launch the kernel
    bmm[grid](
        A, B, O,
        M, N, K,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        O.stride(1), O.stride(2),
        GROUP_M=8,
        DIVISIBLE_M=(M % 16 == 0),
        DIVISIBLE_N=(N % 16 == 0),
        DIVISIBLE_K=(K % 16 == 0),
    )

    return O
