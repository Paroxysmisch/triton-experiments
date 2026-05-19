import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_rowwise_dequantize_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to scaling factors
    state_x_ptr, state_w_ptr,
    # Optional bias pointer
    bias_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Kernel for computing C = dequant(A @ B) where A and B are int8"""
    
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = SPLIT_K
    
    # Get program ID for each dimension
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute tile offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers to A, B, scaling factors
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Load scaling factors for rows
    state_x = tl.load(state_x_ptr + offs_am)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, K, BLOCK_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
        
        # Cast to higher precision for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply row-wise scaling
    acc = acc * state_x[:, None]
    
    # Add bias if provided
    if bias_ptr is not None:
        acc += bias[None, :]
    
    # Store result
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=offs_am[:, None] < M & offs_bn[None, :] < N)

def int8_matmul_rowwise_dequantize(a: torch.Tensor, b: torch.Tensor, 
                                  state_x: torch.Tensor, state_w: torch.Tensor,
                                  bias: torch.Tensor = None) -> torch.Tensor:
    """
    Compute C = dequant(A @ B) where A and B are int8 matrices
    
    Args:
        a: (M, K) int8 matrix
        b: (K, N) int8 matrix
        state_x: (M,) float scaling factors for rows of A
        state_w: (N,) float scaling factors for columns of B
        bias: Optional (N,) bias vector
    
    Returns:
        c: (M, N) float16 matrix
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert state_x.dtype == torch.float32 and state_w.dtype == torch.float32
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    _int8_matmul_rowwise_dequantize_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        state_x_ptr=state_x, state_w_ptr=state_w,
        bias_ptr=bias if bias is not None else None,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
    )
    
    return c
