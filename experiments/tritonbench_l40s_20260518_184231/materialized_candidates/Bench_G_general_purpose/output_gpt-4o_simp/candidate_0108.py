import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M, Out, stride_m, stride_out, SIZE_M, D_HEAD, BLOCK_SIZE: tl.constexpr):
    # Program ID is the unique identifier for each program instance
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Calculate the starting index of each block
    offset_m = pid_m * BLOCK_SIZE
    offset_d = pid_d * BLOCK_SIZE

    # Create pointers to the input and output matrices
    ptr_m = M + offset_m * stride_m + offset_d
    ptr_out = Out + offset_d * stride_out + offset_m

    # Create a block of indices
    m_idx = tl.arange(0, BLOCK_SIZE)
    d_idx = tl.arange(0, BLOCK_SIZE)

    # Load a block from the input matrix
    mask_m = (offset_m + m_idx) < SIZE_M
    mask_d = (offset_d + d_idx) < D_HEAD
    block = tl.load(ptr_m + m_idx[:, None] * stride_m + d_idx[None, :], mask=mask_m[:, None] & mask_d[None, :])

    # Store the transposed block into the output matrix
    tl.store(ptr_out + d_idx[:, None] * stride_out + m_idx[None, :], block, mask=mask_d[:, None] & mask_m[None, :])

def transpose_matrix(M, SIZE_M, D_HEAD):
    # Define block size for tiling
    BLOCK_SIZE = 32

    # Initialize output matrix with zeros
    Out = torch.zeros((D_HEAD, SIZE_M), device='cuda', dtype=M.dtype)

    # Calculate strides
    stride_m = M.stride(0)
    stride_out = Out.stride(0)

    # Launch the kernel
    grid = (triton.cdiv(SIZE_M, BLOCK_SIZE), triton.cdiv(D_HEAD, BLOCK_SIZE))
    kernel[grid](M, Out, stride_m, stride_out, SIZE_M, D_HEAD, BLOCK_SIZE=BLOCK_SIZE)

    return Out

# Example usage
SIZE_M = 128
D_HEAD = 64
M = torch.rand((SIZE_M, D_HEAD), device='cuda', dtype=torch.float32)

# Transpose the matrix
Out = transpose_matrix(M, SIZE_M, D_HEAD)

# Verify correctness
print("Input Matrix (M):")
print(M.cpu().numpy())
print("\nTransposed Matrix (Out):")
print(Out.cpu().numpy())
