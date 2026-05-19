import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, Out_ptr,
    # Matrix dimensions
    dim_m, dim_n, dim_k,
    # Block sizes
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    # Strides
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_om, stride_on,
):
    # Program ID
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    # Output tile indices
    output_m = m_index * block_m + tl.arange(0, block_m)
    output_n = n_index * block_n + tl.arange(0, block_n)

    # Initialize accumulator
    vecmat = tl.zeros((block_m, block_n), dtype=tl.float32)

    # Iterate over k dimension in blocks
    for k in range(0, dim_k, block_k):
        k_index = tl.arange(0, block_k) + k

        # Load and mask A (vector)
        a_mask = k_index < dim_k
        a_ptrs = A_ptr + output_m[:, None] * stride_am + k_index[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load and mask B (matrix)
        b_mask = (output_m[:, None] < dim_m) & (k_index[None, :] < dim_k)
        b_ptrs = B_ptr + output_m[:, None, None] * stride_bm + output_n[None, :] * stride_bn + k_index[None, None, :] * stride_bk
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute dot product
        vecmat += tl.sum(a[:, :, None] * b, axis=1)

    # Store the result
    out_mask = (output_m[:, None] < dim_m) & (output_n[None, :] < dim_n)
    out_ptrs = Out_ptr + output_m[:, None] * stride_om + output_n[None, :] * stride_on
    tl.store(out_ptrs, vecmat, mask=out_mask)

def batched_vecmat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute batched vector-matrix multiplication.
    
    Args:
        A: Input tensor of shape (dim_m, dim_k)
        B: Input tensor of shape (dim_m, dim_n, dim_k)
    
    Returns:
        Output tensor of shape (dim_m, dim_n)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert len(A.shape) == 2 and len(B.shape) == 3, "Invalid input shapes"
    
    dim_m, dim_k = A.shape
    b_dim_m, dim_n, b_dim_k = B.shape
    
    assert dim_m == b_dim_m and dim_k == b_dim_k, "Incompatible dimensions"
    
    # Define block sizes
    block_m = 32
    block_n = 32
    block_k = 32
    
    # Check if dimensions are divisible by block sizes
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m"
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n"
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k"
    
    # Compute grid dimensions
    grid = (dim_m // block_m, dim_n // block_n)
    
    # Initialize output tensor
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)
    
    # Launch kernel
    batched_vecmat_kernel[grid](
        A_ptr=A, B_ptr=B, Out_ptr=output,
        dim_m=dim_m, dim_n=dim_n, dim_k=dim_k,
        block_m=block_m, block_n=block_n, block_k=block_k,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bm=B.stride(0), stride_bn=B.stride(1), stride_bk=B.stride(2),
        stride_om=output.stride(0), stride_on=output.stride(1),
    )
    
    return output
