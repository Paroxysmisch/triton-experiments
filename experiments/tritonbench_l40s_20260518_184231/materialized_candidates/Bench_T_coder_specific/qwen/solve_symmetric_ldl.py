triton
@triton.jit
def ldl_decomposition_kernel(
    A_ptr, L_ptr, D_ptr,
    N: int,
    BLOCK_SIZE: int = 32
):
    pid = triton.program_id(0)
    grid_size = triton.cdiv(N, BLOCK_SIZE)

    row = pid * BLOCK_SIZE + triton.block_idx(0)
    col = triton.block_idx(1)

    if row >= N or col >= N:
        return

    sum_val = 0.0
    for k in range(row):
        sum_val += L_ptr[row * N + k] * L_ptr[col * N + k]
    
    if row == col:
        D_ptr[row * N + col] = A_ptr[row * N + col] - sum_val
        L_ptr[row * N + col] = 1.0
    else:
        L_ptr[row * N + col] = (A_ptr[row * N + col] - sum_val) / D_ptr[col * N + col]

# Triton wrapper function
def solve_symmetric_ldl_triton(A, b, *, hermitian=False, out=None):
    import torch
    from torch.utils.dlpack import to_dlpack, from_dlpack

    # Check input shapes
    assert A.dim() >= 2 and A.shape[-2:] == A.shape[-2:], "A must be a square matrix"
    assert b.dim() >= 1 and A.shape[:-2] == b.shape[:-1], "b shape mismatch"

    # Convert inputs to DLpack format for Triton
    A_dlpack = to_dlpack(A)
    b_dlpack = to_dlpack(b)

    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(b)

    # Convert output tensor to DLpack format
    out_dlpack = to_dlpack(out)

    # Call Triton kernel
    ldl_decomposition_kernel[grid_size, BLOCK_SIZE](A_dlpack, out_dlpack, out_dlpack, A.shape[-1])

    # Convert output back to PyTorch tensor
    return from_dlpack(out_dlpack)
