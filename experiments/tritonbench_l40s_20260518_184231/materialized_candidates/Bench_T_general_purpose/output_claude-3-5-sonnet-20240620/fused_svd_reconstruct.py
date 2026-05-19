import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def fused_svd_reconstruct_kernel(
    # Pointers to matrices
    u_ptr,        # Pointer to U matrix
    s_ptr,        # Pointer to singular values
    vh_ptr,       # Pointer to V^H matrix
    output_ptr,   # Pointer to output matrix
    # Matrix dimensions
    M,           # Number of rows
    N,           # Number of columns
    K,           # Rank (number of singular values)
    stride_um,   # Stride for U matrix
    stride_un,   # Stride for U matrix
    stride_vm,   # Stride for V^H matrix
    stride_vn,   # Stride for V^H matrix
    stride_out_m,# Stride for output matrix
    stride_out_n,# Stride for output matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    block_m = pid // (N // BLOCK_SIZE)
    block_n = pid % (N // BLOCK_SIZE)
    
    # Offsets
    offs_m = block_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = block_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load U block
        u_block_ptr = u_ptr + offs_m[:, None] * stride_um + (k + tl.arange(0, BLOCK_SIZE)[None, :]) * stride_un
        u = tl.load(u_block_ptr, mask=mask_m[:, None], other=0.0)
        
        # Load singular values
        s = tl.load(s_ptr + k + tl.arange(0, BLOCK_SIZE))
        
        # Load V^H block
        vh_block_ptr = vh_ptr + (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_vm + offs_n[None, :] * stride_vn
        vh = tl.load(vh_block_ptr, mask=mask_n[None, :], other=0.0)
        
        # Multiply U * S
        us = u * s[None, :]
        
        # Accumulate US * V^H
        acc += tl.dot(us, vh)
    
    # Store output
    out_ptr = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs matrix A using SVD decomposition: A ≈ U Σ V^H
    
    Args:
        A: Input tensor of shape (M, N)
        
    Returns:
        Reconstructed tensor of shape (M, N)
    """
    M, N = A.shape
    device = A.device
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get dimensions
    K = S.shape[0]  # Number of singular values
    
    # Ensure contiguous matrices
    U = U.contiguous()
    S = S.contiguous()
    Vh = Vh.contiguous()
    
    # Initialize output
    output = torch.empty_like(A)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE) * triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    fused_svd_reconstruct_kernel[grid](
        U,
        S,
        Vh,
        output,
        M,
        N,
        K,
        U.stride(0),
        U.stride(1),
        Vh.stride(0),
        Vh.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
