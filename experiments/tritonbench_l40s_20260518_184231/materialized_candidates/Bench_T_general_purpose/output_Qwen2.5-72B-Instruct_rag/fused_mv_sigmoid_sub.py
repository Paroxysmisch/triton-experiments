import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr,  # Pointer to the input matrix A of shape (n, m)
    vec_ptr,    # Pointer to the input vector v of shape (m)
    other_ptr,  # Pointer to the tensor or scalar b
    out_ptr,    # Pointer to the output tensor
    n,          # Number of rows in the input matrix A
    m,          # Number of columns in the input matrix A
    alpha,      # Scalar multiplier for other
    BLOCK_SIZE: tl.constexpr
):
    # Compute the row index for the current thread
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE

    # Load the input matrix A and the vector v
    input = tl.load(input_ptr + row_start * m, mask=row_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    vec = tl.load(vec_ptr)

    # Perform matrix-vector multiplication
    z = tl.dot(input, vec)

    # Apply the sigmoid activation
    s = 1 / (1 + tl.exp(-z))

    # Load the other tensor or scalar
    other = tl.load(other_ptr, mask=0) * alpha

    # Perform the subtraction
    y = s - other

    # Store the result
    tl.store(out_ptr + row_start, y, mask=row_start + tl.arange(0, BLOCK_SIZE) < n)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n', 'm']
)
@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr,  # Pointer to the input matrix A of shape (n, m)
    vec_ptr,    # Pointer to the input vector v of shape (m)
    other_ptr,  # Pointer to the tensor or scalar b
    out_ptr,    # Pointer to the output tensor
    n,          # Number of rows in the input matrix A
    m,          # Number of columns in the input matrix A
    alpha,      # Scalar multiplier for other
    BLOCK_SIZE: tl.constexpr
):
    # Compute the row index for the current thread
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE

    # Load the input matrix A and the vector v
    input = tl.load(input_ptr + row_start * m, mask=row_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    vec = tl.load(vec_ptr)

    # Perform matrix-vector multiplication
    z = tl.dot(input, vec)

    # Apply the sigmoid activation
    s = 1 / (1 + tl.exp(-z))

    # Load the other tensor or scalar
    other = tl.load(other_ptr, mask=0) * alpha

    # Perform the subtraction
    y = s - other

    # Store the result
    tl.store(out_ptr + row_start, y, mask=row_start + tl.arange(0, BLOCK_SIZE) < n)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Ensure the input and vec are on the same device
    assert input.device == vec.device, "input and vec must be on the same device"
    
    # Ensure the other tensor is a scalar or broadcastable to the output shape
    if isinstance(other, (int, float)):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    else:
        assert other.device == input.device, "other must be on the same device as input"
        assert other.shape == () or other.shape == (input.shape[0],), "other must be a scalar or broadcastable to the output shape"

    # Ensure the shapes are compatible for matrix-vector multiplication
    assert input.shape[1] == vec.shape[0], "input and vec must have compatible shapes for matrix-vector multiplication"

    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty(input.shape[0], device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = (input.shape[0] // 128 + 1,)
    fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out, input.shape[0], input.shape[1], alpha, BLOCK_SIZE=128
    )

    return out

# Example inputs
input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
vec = torch.tensor([1.0, 1.0], device='cuda')
other = torch.tensor(0.5, device='cuda')
alpha = 1.0

# Call the fused operation
output = fused_mv_sigmoid_sub(input, vec, other, alpha)

# Print the output
print(output)
