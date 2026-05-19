import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    # Offsets
    offsets_am = m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize the accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute the pointers to the data
        a_ptrs = a_ptr + (offsets_am[:, None] * stride_am + (k + offsets_k[None, :]) * stride_ak)
        b_ptrs = b_ptr + ((k + offsets_k[:, None]) * stride_bk + offsets_bn[None, :] * stride_bn)
        
        # Load the inputs
        a = tl.load(a_ptrs, mask=offsets_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offsets_bn[None, :] < N, other=0.0)
        
        # Perform the matrix multiplication
        acc += tl.dot(a, b)
    
    # Write back the result
    c_ptrs = c_ptr + (offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=offsets_am[:, None] < M and offsets_bn[None, :] < N)

# The wrapper function
def batched_vecmat(a: torch.Tensor, b: torch.Tensor):
    # Extract the dimensions
    M, K = a.shape
    _, N = b.shape
    
    # Allocate the output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Run the kernel
    batched_vecmat_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    
    return c
