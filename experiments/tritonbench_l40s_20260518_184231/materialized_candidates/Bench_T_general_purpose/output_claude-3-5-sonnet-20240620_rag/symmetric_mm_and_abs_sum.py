import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_kernel(
    # Pointers to matrices
    a_ptr, c_ptr,
    # Matrix dimensions
    M, N,
    # Scaling factors
    alpha, beta,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_an,  # Strides for matrix A
    stride_cm,  # Stride for matrix C (symmetric, so only need one)
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing C = alpha * (A @ A.T) + beta * C
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # Create block pointers for the current block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through k-dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load A block (M, K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_an
        mask_a = (offs_am[:, None] < M) & (k + offs_k[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load A.T block (K, N)
        b_ptrs = a_ptr + (k + offs_k[:, None]) * stride_am + offs_bn[None, :] * stride_an
        mask_b = (k + offs_k[:, None] < N) & (offs_bn[None, :] < M)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Scale by alpha
    acc = acc * alpha
    
    # Load C, scale by beta and add to accumulator
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cm
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < M)
    c = tl.load(c_ptrs, mask=mask_c, other=0.0)
    acc += beta * c
    
    # Write back result
    tl.store(c_ptrs, acc, mask=mask_c)

@triton.jit
def _abs_sum_kernel(
    ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing the sum of absolute values
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and compute absolute values
    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    x = tl.abs(x)
    
    # Compute sum for this block
    block_sum = tl.sum(x, axis=0)
    
    # Write result to output
    tl.atomic_add(ptr, block_sum)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Performs symmetric matrix multiplication and returns absolute sum.
    
    Args:
        A (Tensor): Input matrix of shape (n, m)
        C (Tensor): Matrix of shape (n, n) to accumulate the result
        alpha (float): Scaling factor for the matrix product
        beta (float): Scaling factor for matrix C
    
    Returns:
        Tensor: Scalar tensor with sum of absolute values
    """
    assert A.is_cuda and C.is_cuda, "Input tensors must be on GPU"
    assert A.is_contiguous() and C.is_contiguous(), "Input tensors must be contiguous"
    
    M, N = A.shape
    assert C.shape == (M, M), "C must have shape (n, n) where n is A's first dimension"
    
    # Matrix multiplication configuration
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Grid for matrix multiplication
    grid = lambda meta: (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),
    )
    
    # Compute symmetric matrix multiplication
    _symmetric_mm_kernel[grid](
        A, C,
        M, N,
        alpha, beta,
        A.stride(0), A.stride(1),
        C.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Compute absolute sum
    result = torch.zeros(1, dtype=A.dtype, device=A.device)
    n_elements = M * M
    
    # Grid for reduction
    grid_sum = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Compute sum of absolute values
    _abs_sum_kernel[grid_sum](
        C, n_elements,
        BLOCK_SIZE=1024,
    )
    
    return result
