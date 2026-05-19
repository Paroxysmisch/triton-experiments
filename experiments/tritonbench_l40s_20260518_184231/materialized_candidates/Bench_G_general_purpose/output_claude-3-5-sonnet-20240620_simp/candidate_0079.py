import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_transpose(
    # Pointers to matrices
    a_ptr, b_ptr, absmax_inv_ptr,
    # Matrix dimensions
    M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1 element in that dimension
    stride_am, stride_an,  # stride of A in M and N dimension
    stride_bm, stride_bn,  # stride of B in M and N dimension
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = pid // num_pid_m
    group_id = pid % num_pid_m
    
    # Create block pointers
    offs_am = group_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = num_pid_in_group * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load the scaling factor (inverse of absmax)
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Create mask to handle the case where BLOCK_M is not a multiple of M
    a_mask = offs_am < M
    b_mask = offs_bn < N
    
    # Load input matrix A
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_bn[None, :] * stride_an
    a = tl.load(a_ptrs, mask=a_mask[:, None] & b_mask[None, :], other=0.0)
    
    # Quantize the values
    scaled = a * absmax_inv * 127.0
    quantized = tl.math.round(scaled)
    # Clamp values to int8 range [-127, 127]
    quantized = tl.math.min(tl.math.max(quantized, -127.0), 127.0)
    
    # Transpose and store output matrix B
    # Note the swapped indices for transposition
    b_ptrs = b_ptr + offs_bn[:, None] * stride_bm + offs_am[None, :] * stride_bn
    tl.store(b_ptrs, quantized.to(tl.int8).T, mask=b_mask[:, None] & a_mask[None, :])

def quantize_global_transpose(a: torch.Tensor, absmax_inv: torch.Tensor) -> torch.Tensor:
    """
    Quantize and transpose a matrix using Triton.
    
    Args:
        a: Input tensor of shape (M, N)
        absmax_inv: Inverse of the absolute maximum value for scaling
    
    Returns:
        Quantized and transposed tensor of shape (N, M) with int8 dtype
    """
    assert a.dim() == 2, "Input tensor must be 2-dimensional"
    M, N = a.shape
    
    # Allocate output tensor with transposed dimensions
    b = torch.empty((N, M), device=a.device, dtype=torch.int8)
    
    # Define block sizes (can be tuned for better performance)
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Launch kernel
    _quantize_global_transpose[grid](
        a_ptr=a, 
        b_ptr=b,
        absmax_inv_ptr=absmax_inv,
        M=M, N=N,
        stride_am=a.stride(0), stride_an=a.stride(1),
        stride_bm=b.stride(0), stride_bn=b.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    
    return b
