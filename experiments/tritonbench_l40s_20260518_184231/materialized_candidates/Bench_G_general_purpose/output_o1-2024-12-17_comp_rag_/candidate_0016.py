import triton
import triton.language as tl
import torch
from typing import Optional, Tuple


@triton.autotune(
    configs=[
        triton.Config(
            {
                'TILE_M': 128,
                'TILE_N': 128,
                'TILE_K': 32,
                'GROUP_M': 8,
                'DIVISIBLE_M': True,
                'DIVISIBLE_N': True,
                'DIVISIBLE_K': True,
            },
            num_stages=3, num_warps=4
        ),
        triton.Config(
            {
                'TILE_M': 64,
                'TILE_N': 64,
                'TILE_K': 32,
                'GROUP_M': 4,
                'DIVISIBLE_M': False,
                'DIVISIBLE_N': False,
                'DIVISIBLE_K': False,
            },
            num_stages=2, num_warps=2
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M, N, K,  # dimensions
    strideAm, strideAk,
    strideBk, strideBn,
    strideOm, strideOn,
    batch,  # number of batches
    # Meta-parameters
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,  # determines ordering for compute
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    # Re-map program id to get good locality in M
    # so that programs that work on the same M are executed in groups
    group_size_m = GROUP_M
    group_id = pid_m // group_size_m
    group_pid = pid_m % group_size_m
    pid_m = group_id
    pid_m = group_size_m * pid_m + group_pid

    # Compute the tile's upper-left corner indices
    # in the output (m, n)
    m_start = pid_m * TILE_M
    n_start = pid_n * TILE_N

    # Offsets for loading from A, B, and storing to O
    A_offs = (pid_batch * strideAm * TILE_M)  # shift to batch
    B_offs = (pid_batch * strideBk * TILE_K)  # shift to batch
    O_offs = (pid_batch * strideOm * TILE_M)  # shift to batch

    # Define the range for M, N
    # For partial tiles, we mask loads/stores if needed
    # If dimensions are divisible by tile or if we set the meta
    # then we can skip some boundary checks
    offs_m = m_start + tl.arange(0, TILE_M)
    offs_n = n_start + tl.arange(0, TILE_N)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    else:
        mask_m = True
    if not DIVISIBLE_N:
        mask_n = offs_n < N
    else:
        mask_n = True

    # Pointer arithmetic for the first chunk of A and B
    A_ptrs = A_ptr + A_offs + (offs_m[:, None] * strideAm) + (tl.arange(0, TILE_K)[None, :] * strideAk)
    B_ptrs = B_ptr + B_offs + (tl.arange(0, TILE_K)[:, None] * strideBk) + (offs_n[None, :] * strideBn)

    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K dimension in steps of TILE_K
    # For partial K if not divisible, we use a remainder chunk
    # or rely on the user dimension checks
    k_tiles = (K + TILE_K - 1) // TILE_K if not DIVISIBLE_K else K // TILE_K

    for kt in range(k_tiles):
        # Load the A tile
        a_tile = tl.load(A_ptrs, mask=(mask_m[:, None] & (tl.arange(0, TILE_K)[None, :] + kt*TILE_K < K)), other=0.0)
        # Load the B tile
        b_tile = tl.load(B_ptrs, mask=(mask_n[None, :] & (tl.arange(0, TILE_K)[:, None] + kt*TILE_K < K)), other=0.0)
        # Compute partial product
        acc += tl.dot(a_tile, b_tile)

        # Advance pointers to the next K-slice
        A_ptrs += TILE_K * strideAk
        B_ptrs += TILE_K * strideBk

    # Compute O tile base address
    O_ptrs = O_ptr + O_offs + (offs_m[:, None] * strideOm) + (offs_n[None, :] * strideOn)

    # Store the result
    tl.store(O_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    """
    Batched matrix multiplication (A @ B) -> out using Triton.
    A: (batch, M, K)
    B: (batch, K, N)
    out: (batch, M, N)
    """
    assert A.dim() == 3 and B.dim() == 3, "A and B must be rank-3 tensors."
    batch, M, K = A.shape
    assert B.shape[0] == batch and B.shape[1] == K, "B must match batch and K dimension."
    N = B.shape[2]

    # Allocate output if none provided
    if out is None:
        out = A.new_empty((batch, M, N))

    # Make sure tensors are on the same device
    device = A.device
    assert B.device == device and out.device == device, "All tensors must be on the same device."

    # Launch kernel
    grid = lambda META: (
        (M + META['TILE_M'] - 1) // META['TILE_M'] * META['GROUP_M'],
        (N + META['TILE_N'] - 1) // META['TILE_N'],
        batch
    )

    bmm_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        out.stride(1), out.stride(2),
        batch,
    )

    return out
