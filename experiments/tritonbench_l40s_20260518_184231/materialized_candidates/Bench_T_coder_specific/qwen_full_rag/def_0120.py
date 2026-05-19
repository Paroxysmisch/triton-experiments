import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_dot_kernel(A, x, y, alpha, beta):
    # Compute the matrix-vector product and update y
    y *= beta
    chunk_size = 4
    n = A.shape[0]
    m = A.shape[1]
    pid = tl.program_id(0)
    block_start = pid * chunk_size
    offsets = block_start + tl.arange(0, chunk_size)
    mask = offsets < n
    a = tl.load(A + offsets[:, None] * m, mask=mask, other=0.0)
    x_broadcasted = tl.broadcast_to(x, a)
    y += tl.sum(a * x_broadcasted, axis=1)

    # Calculate the dot product of the updated y with x
    result = tl.sum(y * x)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Wrapper function for the Triton kernel
    assert A.shape[1] == x.shape[0] and A.shape[0] == y.shape[0], "Incompatible shapes"
    grid = lambda meta: (triton.cdiv(A.shape[0], meta["chunk_size"]),)
    matrix_vector_dot_kernel[grid](A, x, y, alpha, beta)
    return y
