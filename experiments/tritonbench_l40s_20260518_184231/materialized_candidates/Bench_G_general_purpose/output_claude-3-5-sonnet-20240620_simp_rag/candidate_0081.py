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
    # Optional bias pointer
    bias_ptr = None,
):
    """Kernel for computing the matmul C = A x B.
    A is of shape (M, K) float16
    B is of shape (K, N) float16
    C is of shape (M, N) float16
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = SPLIT_K
    num_pid_in_group = num_pid_n * num_pid_m
    group_id = pid // num_pid_in_group
    pid_m = (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # Load scaling factors for current row
    offs_scale_x = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    state_x_scale = tl.load(state_x_ptr + offs_scale_x, mask=offs_scale_x < M)

    # Block pointers
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the scaling factor for weights
        state_w_scale = tl.load(state_w_ptr + k * BLOCK_K + offs_k)
        
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))
        
        # Dequantize A and B using scaling factors
        a = (a.to(tl.float32) * state_x_scale[:, None])
        b = (b.to(tl.float32) * state_w_scale[:, None])
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store results
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_cn)
        acc += bias[None, :]
    
    # Store output
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

def int8_matmul_rowwise_dequantize(a: torch.Tensor, b: torch.Tensor, 
                                  state_x: torch.Tensor, state_w: torch.Tensor,
                                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute matrix multiplication C = A × B with row-wise dequantization
    Args:
        a: Quantized input matrix (M, K) in int8
        b: Quantized weight matrix (K, N) in int8
        state_x: Row-wise scaling factors for input matrix (M,)
        state_w: Row-wise scaling factors for weight matrix (K,)
        bias: Optional bias tensor (N,)
    Returns:
        c: Output matrix (M, N) in float16
    """
    # Check constraints
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} @ {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']) * META['SPLIT_K'],
    )
    
    _int8_matmul_rowwise_dequantize_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        state_x_ptr=state_x, state_w_ptr=state_w,
        M=M, N=N, K=K,
        stride_am=K, stride_ak=1,
        stride_bk=N, stride_bn=1,
        stride_cm=N, stride_cn=1,
        bias_ptr=bias if bias is not None else None,
    )
    
    return c
