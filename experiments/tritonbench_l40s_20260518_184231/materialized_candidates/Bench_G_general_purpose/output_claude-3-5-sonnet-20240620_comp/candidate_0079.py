import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_transpose_kernel(
    # Pointers to matrices
    A, B,
    # Matrix dimensions
    M, N,
    # Strides for accessing matrices
    stride_am, stride_an,
    stride_bm, stride_bn,
    # Quantization parameter
    absmax_inv,
    # Block level metadata
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Number of blocks in N dimension
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Calculate group and block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block start indices
    block_m_start = pid_m * BLOCK_M
    block_n_start = pid_n * BLOCK_N
    
    # Group offset calculation
    group_m = block_m_start // GROUP_M
    
    # Create offsets for this block
    offs_m = block_m_start + tl.arange(0, BLOCK_M)
    offs_n = block_n_start + tl.arange(0, BLOCK_N)
    
    # Create mask for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Compute input memory addresses
    a_ptrs = A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    
    # Load input data
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Quantize to int8 range (-127 to 127)
    a_quantized = (a * absmax_inv * 127.0).to(tl.int8)
    
    # Compute output memory addresses (with transpose)
    b_ptrs = B + offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn
    
    # Store transposed and quantized result
    tl.store(b_ptrs, a_quantized.T, mask=mask_n[:, None] & mask_m[None, :])

def quantize_global_transpose(A: torch.Tensor) -> torch.Tensor:
    """
    Quantize and transpose a matrix using Triton.
    
    Args:
        A: Input tensor of shape (M, N)
    Returns:
        Quantized and transposed tensor of shape (N, M) in int8 format
    """
    assert A.dim() == 2, "Input tensor must be 2-dimensional"
    M, N = A.shape
    
    # Calculate absolute maximum for quantization
    absmax = torch.max(torch.abs(A))
    absmax_inv = 1.0 / absmax.item()
    
    # Create output tensor
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    GROUP_M = 64
    
    # Calculate grid size
    def grid(meta):
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Launch kernel
    _quantize_global_transpose_kernel[grid](
        A, B,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        absmax_inv,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
    )
    
    return B

# Example usage
if __name__ == "__main__":
    # Create sample input
    M, N = 1024, 2048
    A = torch.randn(M, N, device='cuda')
    
    # Run quantization and transposition
    B = quantize_global_transpose(A)
    
    print(f"Input shape: {A.shape}")
    print(f"Output shape: {B.shape}")
    print(f"Output dtype: {B.dtype}")
