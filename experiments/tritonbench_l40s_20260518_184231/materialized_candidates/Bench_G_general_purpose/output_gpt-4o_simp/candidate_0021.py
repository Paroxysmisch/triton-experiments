import triton
import triton.language as tl

# Define the kernel for batched matrix multiplication
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    # Define the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Compute the starting indices of the tiles
    offs_am = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_bn = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Initialize the accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, TILE_K):
        # Load tiles from A and B
        A_tile = tl.load(A_ptr + pid_batch * stride_am * M + offs_am[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak, mask=(offs_am[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        B_tile = tl.load(B_ptr + pid_batch * stride_bk * K + (k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=((k + offs_k)[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        # Accumulate the results
        acc += tl.dot(A_tile, B_tile)

    # Store the result
    O_ptr += pid_batch * stride_om * M + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    tl.store(O_ptr, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

# Wrapper function for batched matrix multiplication
def bmm(A, B, M, N, K):
    # Get the batch size
    batch = A.shape[0]

    # Define the tile sizes
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32

    # Allocate output tensor
    O = torch.empty((batch, M, N), device='cuda', dtype=torch.float32)

    # Launch the kernel
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), batch)
    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        O.stride(1), O.stride(2),
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K
    )

    return O
