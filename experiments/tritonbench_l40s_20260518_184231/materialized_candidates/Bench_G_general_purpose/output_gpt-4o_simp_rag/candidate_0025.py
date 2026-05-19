import triton
import triton.language as tl
import torch

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    # Offsets for the current tile
    offs_am = pid * TILE_M + tl.arange(0, TILE_M)
    offs_bn = pid * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Pointers to the start of the batch
    A_ptr = A_ptr + batch_id * M * K
    B_ptr = B_ptr + batch_id * K * N
    O_ptr = O_ptr + batch_id * M * N

    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, TILE_K):
        # Load tiles from A and B
        A_tile = tl.load(A_ptr + offs_am[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak, mask=offs_am[:, None] < M)
        B_tile = tl.load(B_ptr + (k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=offs_bn[None, :] < N)

        # Accumulate matrix multiplication
        acc += tl.dot(A_tile, B_tile)

    # Store the result in O
    O_tile = O_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    tl.store(O_tile, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def bmm(A, B, M, N, K, TILE_M=128, TILE_N=128, TILE_K=32):
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    batch_size = A.shape[0]

    # Allocate output tensor
    O = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    # Strides for the input and output tensors
    stride_am, stride_ak = A.stride(1), A.stride(2)
    stride_bk, stride_bn = B.stride(1), B.stride(2)
    stride_om, stride_on = O.stride(1), O.stride(2)

    # Launch kernel
    grid = (triton.cdiv(M, TILE_M), batch_size)
    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K
    )

    return O
