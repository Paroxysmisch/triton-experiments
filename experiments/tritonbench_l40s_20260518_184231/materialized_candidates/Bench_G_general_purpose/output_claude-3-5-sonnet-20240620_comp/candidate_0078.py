import triton
import triton.language as tl
import torch

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
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    """Kernel for computing int8 matrix multiplication C = A @ B with row-wise dequantization"""
    
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = SPLIT_K
    
    # Program ID to block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block start indices
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load scaling factors for current block
        scale_x = tl.load(state_x_ptr + offs_am)
        scale_w = tl.load(state_w_ptr + k * BLOCK_K + offs_k)
        
        # Compute pointer offsets for A and B
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k * BLOCK_K + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k * BLOCK_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        
        # Load data
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and (k * BLOCK_K + offs_k[None, :]) < K)
        b = tl.load(b_ptrs, mask=(k * BLOCK_K + offs_k[:, None]) < K and offs_bn[None, :] < N)
        
        # Matrix multiply
        acc += tl.dot(a, b)
    
    # Dequantize
    acc = acc.to(tl.float32)
    acc = acc * scale_x[:, None] * scale_w[None, :]
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
        acc += bias[None, :]
    
    # Store output
    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    # Write back output
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)

# Wrapper function
def int8_matmul_rowwise_dequantize(a: torch.Tensor, b: torch.Tensor, 
                                  state_x: torch.Tensor, state_w: torch.Tensor,
                                  bias: torch.Tensor = None,
                                  BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 32,
                                  GROUP_M: int = 8, SPLIT_K: int = 1) -> torch.Tensor:
    """
    Compute int8 matrix multiplication with row-wise dequantization: C = A @ B
    
    Args:
        a: Input matrix A (M, K) - int8
        b: Input matrix B (K, N) - int8
        state_x: Row-wise scaling factors for A (M,) - float32
        state_w: Row-wise scaling factors for B (K,) - float32
        bias: Optional bias tensor (N,) - float32
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
        GROUP_M: Number of groups for M dimension
        SPLIT_K: Number of splits in K dimension
    """
    # Check input constraints
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert state_x.dtype == torch.float32 and state_w.dtype == torch.float32
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {K} != {K_}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    _int8_matmul_rowwise_dequantize_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        state_x_ptr=state_x, state_w_ptr=state_w,
        bias_ptr=bias if bias is not None else None,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, SPLIT_K=SPLIT_K,
    )
    
    return c
