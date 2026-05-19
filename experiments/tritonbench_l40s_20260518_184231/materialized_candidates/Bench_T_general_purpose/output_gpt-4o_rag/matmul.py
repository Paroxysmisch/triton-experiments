import triton
import triton.language as tl
import torch

# Triton kernel for matrix-matrix product
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    col_start = tl.arange(0, BLOCK_SIZE)

    # Load blocks of A and B
    A_block = tl.load(A_ptr + row_start * K + col_start[:, None], mask=(row_start + col_start[:, None] < M))
    B_block = tl.load(B_ptr + col_start * N + tl.arange(0, BLOCK_SIZE)[None, :], mask=(col_start + tl.arange(0, BLOCK_SIZE)[None, :] < N))

    # Compute block of C
    C_block = tl.dot(A_block, B_block)

    # Store block of C
    tl.store(C_ptr + row_start * N + col_start[:, None], C_block, mask=(row_start + col_start[:, None] < M))

def matmul(input, other, *, out=None):
    if input.dim() == 1 and other.dim() == 1:
        # Dot product case
        if input.size(0) != other.size(0):
            raise ValueError("Input tensors must be of the same size for dot product.")
        out = torch.dot(input, other) if out is None else out.copy_(torch.dot(input, other))
    elif input.dim() == 2 and other.dim() == 2:
        # Matrix-matrix product
        M, K = input.shape
        K_, N = other.shape
        if K != K_:
            raise ValueError("Incompatible dimensions for matrix multiplication.")
        if out is None:
            out = torch.empty((M, N), dtype=input.dtype, device=input.device)
        BLOCK_SIZE = 128  # Example block size
        grid = (M // BLOCK_SIZE,)
        matmul_kernel[grid](input, other, out, M, N, K, BLOCK_SIZE)
    elif input.dim() == 1 and other.dim() == 2:
        # Matrix-vector product
        out = torch.mv(other, input) if out is None else out.copy_(torch.mv(other, input))
    elif input.dim() == 2 and other.dim() == 1:
        # Vector-matrix product
        out = torch.mv(input, other) if out is None else out.copy_(torch.mv(input, other))
    else:
        # Batched matrix multiplication
        out = torch.bmm(input, other) if out is None else out.copy_(torch.bmm(input, other))

    return out
