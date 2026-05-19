import triton
import triton.language as tl
import torch

@triton.jit
def bmm_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, O_ptr,
    # Matrix dimensions
    BATCH, M, N, K,
    # Strides for A, B, O
    stride_Abatch, stride_Am, stride_Ak,
    stride_Bbatch, stride_Bk, stride_Bn,
    stride_Obatch, stride_Om, stride_On,
    # Tile sizes and configuration
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr,
    # Other params
    ACC_DTYPE: tl.constexpr, ALLOW_TF32: tl.constexpr,
):
    # Map program IDs to tile coordinates
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the current tile
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    
    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=ACC_DTYPE)

    # Load A and B tiles and compute acc
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_remaining = K - k * TILE_K
        a = tl.load(
            A_ptr + pid_batch * stride_Abatch + offs_m[:, None] * stride_Am + (k * TILE_K + offs_k[None, :]) * stride_Ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining) if not DIVISIBLE_M or not DIVISIBLE_K else None,
            other=0.0
        )
        b = tl.load(
            B_ptr + pid_batch * stride_Bbatch + (k * TILE_K + offs_k[:, None]) * stride_Bk + offs_n[None, :] * stride_Bn,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N) if not DIVISIBLE_K or not DIVISIBLE_N else None,
            other=0.0
        )
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_DTYPE)
        k += TILE_K

    # Write back result
    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    tl.store(
        O_ptr + pid_batch * stride_Obatch + offs_om[:, None] * stride_Om + offs_on[None, :] * stride_On,
        acc,
        mask=(offs_om[:, None] < M) & (offs_on[None, :] < N) if not DIVISIBLE_M or not DIVISIBLE_N else None
    )

def bmm(
    A: torch.Tensor, 
    B: torch.Tensor, 
    out: Optional[torch.Tensor] = None, 
    allow_tf32: bool = True,
    autotune: bool = True,
) -> torch.Tensor:
    assert A.shape[0] == B.shape[0], "Batch size mismatch"
    assert A.shape[2] == B.shape[1], "Matrix dimensions mismatch"
    BATCH, M, K = A.shape
    _, _, N = B.shape
    
    # Initialize output tensor
    if out is None:
        out = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)
    else:
        assert out.shape == (BATCH, M, N), "Invalid output shape"
    
    # Define autotune configurations
    configs = [
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 16, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 16, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 16, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 128, 'TILE_K': 16, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ]
    
    # Helper function to compute kernel parameters
    def grid(META):
        return (BATCH, triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)
    
    # Kernel selection with autotuning
    best_config = None
    if autotune:
        best_time = float('inf')
        for config in configs:
            divisible_m = (M % config.kwargs['TILE_M']) == 0
            divisible_n = (N % config.kwargs['TILE_N']) == 0
            divisible_k = (K % config.kwargs['TILE_K']) == 0
            try:
                with torch.cuda.device(A.device):
                    bmm_kernel[grid(config.kwargs)](
                        A, B, out,
                        BATCH, M, N, K,
                        A.stride(0), A.stride(1), A.stride(2),
                        B.stride(0), B.stride(1), B.stride(2),
                        out.stride(0), out.stride(1), out.stride(2),
                        DIVISIBLE_M=divisible_m,
                        DIVISIBLE_N=divisible_n,
                        DIVISIBLE_K=divisible_k,
                        ACC_DTYPE=tl.float32 if out.dtype == torch.float32 else tl.float16,
                        ALLOW_TF32=allow_tf32,
                        **config.kwargs
                    )
                    torch.cuda.synchronize()
                    # Time measurement logic here (omitted for brevity)
                    # Update best_config if this config is faster
            except triton.OutOfResources:
                continue
        # Use best_config for final launch
    
    # Launch kernel with best configuration
    best_config = configs[0]  # Placeholder, replace with actual best config
    divisible_m = (M % best_config.kwargs['TILE_M']) == 0
    divisible_n = (N % best_config.kwargs['TILE_N']) == 0
    divisible_k = (K % best_config.kwargs['TILE_K']) == 0
    
    bmm_kernel[grid(best_config.kwargs)](
        A, B, out,
        BATCH, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        DIVISIBLE_M=divisible_m,
        DIVISIBLE_N=divisible_n,
        DIVISIBLE_K=divisible_k,
        ACC_DTYPE=tl.float32 if out.dtype == torch.float32 else tl.float16,
        ALLOW_TF32=allow_tf32,
        **best_config.kwargs
    )
    
    return out
