import torch

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.shape == (A.size(0), A.size(1))
    assert B.shape == (A.size(1), B.size(1))
    assert C.shape == (A.size(0), B.size(1))
    assert C.numel() > 0, "C must have at least one element"
    
    # Get device and dtype
    device = A.device
    dtype = A.dtype
    
    # Create output tensor
    C_out = torch.empty_like(C, device=device, dtype=dtype)
    
    # Triton grid size and block size
    grid_size = (C_out.size(0) + 64 - 1) // 64
    block_size = 64
    
    # Launch Triton kernel
    matmul_add_kernel[grid_size, block_size](
        A.contiguous().data_ptr(), B.contiguous().data_ptr(), C_out.contiguous().data_ptr(),
        A.size(0), A.size(1), B.size(1), C_out.size(1),
        alpha, beta,
        A.stride(1), B.stride(1), C_out.stride(1),
        BLOCK_SIZE=block_size
    )
    
    # Compute dot product of the first two rows
    result = torch.dot(C_out[0], C_out[1])
    
    return result
