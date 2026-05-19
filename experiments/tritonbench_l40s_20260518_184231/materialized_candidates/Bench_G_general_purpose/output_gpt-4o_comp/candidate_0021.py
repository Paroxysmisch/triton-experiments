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
    pid = tl.program_id(0)
    
    # Calculate grid indices
    batch_id = pid // GROUP_M
    m_tile_id = pid % GROUP_M
    
    # Calculate starting indices for each tile
    offs_m = m_tile_id * TILE_M + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    
    # Initialize accumulators
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, TILE_K):
        # Load tiles of A and B
        a = tl.load(A + batch_id * M * K + (offs_m[:, None] * stride_am + (k + offs_k) * stride_ak), mask=offs_m[:, None] < M if not DIVISIBLE_M else None, other=0.0)
        b = tl.load(B + batch_id * K * N + ((k + offs_k)[:, None] * stride_bk + offs_n * stride_bn), mask=offs_n < N if not DIVISIBLE_N else None, other=0.0)
        
        # Perform matrix multiplication and accumulate
        acc += tl.dot(a, b)
    
    # Write back the results
    c = acc.to(tl.float32)
    tl.store(O + batch_id * M * N + offs_m[:, None] * stride_om + offs_n * stride_on, c, mask=(offs_m[:, None] < M) & (offs_n < N) if not (DIVISIBLE_M and DIVISIBLE_N) else None)

def bmm(A, B, M, N, K, device):
    # Initialize output tensor
    O = torch.empty((A.shape[0], M, N), device=device, dtype=torch.float32)
    
    # Determine grid dimensions
    grid = lambda META: (A.shape[0] * (M + META['TILE_M'] - 1) // META['TILE_M'],)
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        O.stride(1), O.stride(2),
        TILE_M=128, TILE_N=128, TILE_K=32,  # Example tile sizes
        GROUP_M=8,
        DIVISIBLE_M=(M % 128 == 0), DIVISIBLE_N=(N % 128 == 0), DIVISIBLE_K=(K % 32 == 0),
        num_warps=4,
        num_stages=3,
    )
    
    return O
