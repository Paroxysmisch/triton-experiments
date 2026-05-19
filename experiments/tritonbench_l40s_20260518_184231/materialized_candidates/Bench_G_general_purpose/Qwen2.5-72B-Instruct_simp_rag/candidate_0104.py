import triton
import triton.language as tl
import torch

# Define the kernel function for matrix transposition
@triton.jit
def kernel(M_ptr, Out_ptr, stride_m, stride_d, size_m, d_head, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_d = tl.arange(0, d_head)

    mask = (offsets_m < size_m)[:, None]

    M_offsets = offsets_m[:, None] * stride_m + offsets_d[None, :] * stride_d
    Out_offsets = offsets_d[None, :] * stride_m + offsets_m[:, None] * stride_d

    M = tl.load(M_ptr + M_offsets, mask=mask, other=0.0)
    tl.store(Out_ptr + Out_offsets, M, mask=mask)

# Define the wrapper function to initialize matrices and launch the kernel
def transpose_matrix(M, D_HEAD):
    SIZE_M, _ = M.shape
    Out = torch.zeros((D_HEAD, SIZE_M), device=M.device, dtype=M.dtype)

    grid = lambda META: (triton.cdiv(SIZE_M, META['BLOCK_SIZE']),)
    kernel[grid](M, Out, M.stride(0), M.stride(1), SIZE_M, D_HEAD, BLOCK_SIZE=128)

    return Out

# Example usage
if __name__ == "__main__":
    SIZE_M = 1024
    D_HEAD = 64

    # Initialize the input matrix with random values
    M = torch.randn((SIZE_M, D_HEAD), device='cuda', dtype=torch.float32)

    # Transpose the matrix
    Out = transpose_matrix(M, D_HEAD)

    # Print the shapes to verify
    print("Input matrix shape:", M.shape)
    print("Output matrix shape:", Out.shape)
