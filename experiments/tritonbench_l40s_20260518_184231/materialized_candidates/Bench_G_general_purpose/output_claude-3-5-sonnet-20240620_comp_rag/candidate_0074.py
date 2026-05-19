import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_transpose(
    # Pointers to matrices
    A, B,
    # Matrix dimensions
    M, N,
    # Strides for accessing matrices
    stride_am, stride_an,
    stride_bm, stride_bn,
    # Quantization parameter
    absmax_inv,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Number of blocks in N dimension
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Calculate pid_m and pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Calculate offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_an = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create mask for bounds checking
    mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)
    
    # Load data from A
    a = tl.load(A + offs_am[:, None] * stride_am + offs_an[None, :] * stride_an, mask=mask)
    
    # Quantize to int8 range
    a = a * absmax_inv
    a = tl.math.round(a * 127.0)
    a = tl.math.clamp(a, -128.0, 127.0)
    
    # Store transposed and quantized result in B
    # Note the swapped indices for transposition
    tl.store(B + offs_an[:, None] * stride_bm + offs_am[None, :] * stride_bn,
             a.to(tl.int8), mask=mask.T)

def quantize_global_transpose(A, absmax=None):
    # Get matrix dimensions
    M, N = A.shape
    
    # Calculate or use provided absmax
    if absmax is None:
        absmax = torch.max(torch.abs(A))
    absmax_inv = 1.0 / absmax
    
    # Initialize output matrix (transposed dimensions)
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    GROUP_M = 8
    
    # Calculate grid dimensions
    def grid(meta):
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Launch kernel
    _quantize_global_transpose[grid](
        A, B,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        absmax_inv,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
    )
    
    return B, absmax
