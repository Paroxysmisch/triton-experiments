import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M_ptr, Out_ptr, stride_m, stride_out, SIZE_M, D_HEAD, BLOCK_SIZE: tl.constexpr):
    # Program ID corresponds to the block index
    pid = tl.program_id(axis=0)
    
    # Calculate the start of the block in both dimensions
    block_start_m = pid * BLOCK_SIZE
    block_start_d = tl.arange(0, BLOCK_SIZE)
    
    # Load the input block
    input_block = tl.load(M_ptr + block_start_m[:, None] * stride_m + block_start_d[None, :])
    
    # Transpose the block
    transposed_block = tl.trans(input_block)
    
    # Store the transposed block to the output matrix
    tl.store(Out_ptr + block_start_d[:, None] * stride_out + block_start_m[None, :], transposed_block)


def transpose_matrix(M, SIZE_M, D_HEAD, BLOCK_SIZE=16):
    # Initialize output matrix
    Out = torch.zeros((D_HEAD, SIZE_M), device=M.device, dtype=M.dtype)
    
    # Strides for the input and output matrices
    stride_m = M.stride(0)
    stride_out = Out.stride(0)
    
    # Launch the kernel
    grid = lambda META: (triton.cdiv(SIZE_M, BLOCK_SIZE),)
    kernel[grid](M, Out, stride_m, stride_out, SIZE_M, D_HEAD, BLOCK_SIZE=BLOCK_SIZE)
    
    return Out


# Example usage
SIZE_M, D_HEAD = 128, 256
M = torch.randn((SIZE_M, D_HEAD), device='cuda', dtype=torch.float32)
Out = transpose_matrix(M, SIZE_M, D_HEAD)
