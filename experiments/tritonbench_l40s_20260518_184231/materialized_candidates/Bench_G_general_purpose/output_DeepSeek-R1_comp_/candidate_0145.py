import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_warps=8),
    ],
    key=['K', 'N']
)
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_b_k, stride_b_n,
    stride_fpb_k, stride_fpb_n,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_off = pid_n * BLOCK_SIZE_N
    k_off = pid_k * BLOCK_SIZE_K

    # Create masks to handle boundary conditions
    n_mask = tl.arange(0, BLOCK_SIZE_N) < N - n_off
    k_mask = tl.arange(0, BLOCK_SIZE_K) < K - k_off
    block_mask = k_mask[:, None] & n_mask[None, :]

    # Pointers for the current block in B matrix
    b_block_ptr = b_ptr + k_off * stride_b_k + n_off * stride_b_n
    b_block = tl.load(b_block_ptr, mask=block_mask, other=0)

    # Pointers and data for scales (per-column scaling)
    scale_ptr = b_scale_ptr + n_off
    scales = tl.load(scale_ptr, mask=n_mask, other=0)

    # Dequantize: int8 -> float32 and multiply by scales
    fpb_block = b_block.to(tl.float32) * scales[None, :]

    # Store dequantized block to output
    fpb_block_ptr = fpb_ptr + k_off * stride_fpb_k + n_off * stride_fpb_n
    tl.store(fpb_block_ptr, fpb_block, mask=block_mask)

def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor):
    # Check input dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Dimension mismatch between a and b"
    assert b_scale.shape == (N,), "b_scale must have shape (N,)"

    # Allocate output matrix for dequantized b
    fpb = torch.empty((K, N), device=b.device, dtype=torch.float32)

    # 2D launch grid for dequantization kernel
    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
        triton.cdiv(K, meta['BLOCK_SIZE_K']),
    )

    # Get strides for B and FPb matrices
    stride_b_k = b.stride(0)
    stride_b_n = b.stride(1)
    stride_fpb_k = fpb.stride(0)
    stride_fpb_n = fpb.stride(1)

    # Launch dequantization kernel
    dequantize_kernel[grid](
        b, b_scale, fpb,
        K, N,
        stride_b_k, stride_b_n,
        stride_fpb_k, stride_fpb_n
    )

    # Perform matrix multiplication
    return torch.mm(a, fpb)

# Example usage
M, K, N = 512, 1024, 2048

# Create random inputs (int8 quantized)
a = torch.randn(M, K, device='cuda', dtype=torch.float32)
b = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
b_scale = torch.randn(N, device='cuda')

# Perform dequantization and matrix multiplication
result = matmul_dequantize_int8(a, b, b_scale)
