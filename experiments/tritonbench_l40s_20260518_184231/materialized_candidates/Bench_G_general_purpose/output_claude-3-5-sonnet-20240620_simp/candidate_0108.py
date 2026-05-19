import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M_ptr,  # Pointer to input matrix
    Out_ptr,  # Pointer to output matrix
    stride_m_m,  # Stride for M matrix rows
    stride_m_k,  # Stride for M matrix columns
    stride_out_m,  # Stride for Out matrix rows
    stride_out_k,  # Stride for Out matrix columns
    SIZE_M,  # Size of first dimension
    D_HEAD,  # Size of second dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Block start offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Create mask for bounds checking
    mask_m = offs_m < SIZE_M
    mask_k = offs_k < D_HEAD

    # Compute input offsets and load data
    m_ptrs = M_ptr + offs_m[:, None] * stride_m_m + offs_k[None, :] * stride_m_k
    matrix = tl.load(m_ptrs, mask=mask_m[:, None] & mask_k[None, :])

    # Compute output offsets and store transposed data
    out_ptrs = Out_ptr + offs_k[:, None] * stride_out_m + offs_m[None, :] * stride_out_k
    tl.store(out_ptrs, tl.trans(matrix), mask=mask_k[:, None] & mask_m[None, :])

def transpose_matrix(M, SIZE_M, D_HEAD):
    # Ensure input is on GPU
    M = M.cuda()
    
    # Create output tensor
    Out = torch.empty((D_HEAD, SIZE_M), device='cuda', dtype=M.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32

    # Calculate grid dimensions
    grid = (
        triton.cdiv(SIZE_M, BLOCK_SIZE_M),
        triton.cdiv(D_HEAD, BLOCK_SIZE_K)
    )

    # Launch kernel
    kernel[grid](
        M_ptr=M,
        Out_ptr=Out,
        stride_m_m=M.stride(0),
        stride_m_k=M.stride(1),
        stride_out_m=Out.stride(0),
        stride_out_k=Out.stride(1),
        SIZE_M=SIZE_M,
        D_HEAD=D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return Out

# Example usage
if __name__ == "__main__":
    SIZE_M = 1024
    D_HEAD = 64
    
    # Create input matrix with random values
    M = torch.randn((SIZE_M, D_HEAD), device='cuda')
    
    # Transpose using our Triton kernel
    result = transpose_matrix(M, SIZE_M, D_HEAD)
    
    # Verify result
    torch_result = M.T
    assert torch.allclose(result, torch_result, rtol=1e-3, atol=1e-3)
    print("Transposition successful! Results match PyTorch's implementation.")
