import triton
import triton.language as tl
import torch

@triton.jit
def mv_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_an,  
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Row index
    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Iterate through the columns
    for n in range(0, N, BLOCK_SIZE_N):
        # Create block pointers
        a_block_ptr = a_ptr + row_idx[:, None] * stride_am + (n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_an
        b_block_ptr = b_ptr + (n + tl.arange(0, BLOCK_SIZE_N))
        
        # Load data
        mask = n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N
        a = tl.load(a_block_ptr, mask=mask, other=0.0)
        b = tl.load(b_block_ptr, mask=n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
        
        # Compute matrix-vector product for this block
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Write back result
    mask_m = row_idx < M
    tl.store(c_ptr + row_idx, acc, mask=mask_m)

def mv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix-vector product C = A @ B
    
    Parameters:
        a: torch.Tensor of shape (M, N)
        b: torch.Tensor of shape (N,)
    
    Returns:
        c: torch.Tensor of shape (M,)
    """
    assert a.dim() == 2 and b.dim() == 1, "Incorrect input dimensions"
    M, N = a.shape
    assert b.shape[0] == N, "Incompatible dimensions"
    
    # Allocate output
    c = torch.empty(M, device=a.device, dtype=a.dtype)
    
    # Define meta-parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Grid dimension
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    
    # Launch kernel
    mv_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        M=M, N=N,
        stride_am=a.stride(0),
        stride_an=a.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return c
