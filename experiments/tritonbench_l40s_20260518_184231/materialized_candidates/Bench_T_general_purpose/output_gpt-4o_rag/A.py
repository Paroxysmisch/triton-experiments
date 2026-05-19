import torch
import triton
import triton.language as tl

@triton.jit
def solve_kernel(A_ptr, B_ptr, out_ptr, n, batch_stride_A, batch_stride_B, batch_stride_out, left: tl.constexpr):
    # Get the program ID and stride for batch processing
    batch_id = tl.program_id(0)
    A = A_ptr + batch_id * batch_stride_A
    B = B_ptr + batch_id * batch_stride_B
    out = out_ptr + batch_id * batch_stride_out

    # Create pointers for rows and columns
    row_idx = tl.arange(0, n)
    col_idx = tl.arange(0, n)

    # Load matrix A and B
    A_matrix = tl.load(A + row_idx[:, None] * n + col_idx[None, :])
    B_matrix = tl.load(B + row_idx[:, None] * n + col_idx[None, :])

    # Invert matrix A
    A_inv = tl.lib.linalg.inv(A_matrix)

    # Compute the solution
    if left:
        X = tl.lib.linalg.matmul(A_inv, B_matrix)
    else:
        X = tl.lib.linalg.matmul(B_matrix, A_inv)

    # Store the result
    tl.store(out + row_idx[:, None] * n + col_idx[None, :], X)

def solve(A, B, *, left=True, out=None):
    assert A.ndim >= 2 and A.shape[-1] == A.shape[-2], "A must be a square matrix or batch of square matrices"
    assert B.ndim >= 2, "B must be at least a 2D tensor"
    assert A.shape[:-2] == B.shape[:-2], "Batch dimensions of A and B must match"
    assert A.shape[-1] == B.shape[-2], "The inner dimensions of A and B must match"

    # Determine the shape
    batch_size = A.shape[:-2]
    n = A.shape[-1]

    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(B)

    # Define grid size for Triton kernel
    grid = (torch.prod(torch.tensor(batch_size)).item(),)

    # Launch the Triton kernel
    solve_kernel[grid](
        A,
        B,
        out,
        n,
        A.stride(0),
        B.stride(0),
        out.stride(0),
        left=left
    )

    return out

# Example usage
A = torch.randn((2, 3, 3), device='cuda', dtype=torch.float32)
B = torch.randn((2, 3, 3), device='cuda', dtype=torch.float32)
result = solve(A, B)
print(result)
