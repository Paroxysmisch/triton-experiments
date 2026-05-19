import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,  # input/output tensor
    bias_ptr,  # bias tensor
    in_ptr,  # input tensor for multiplication
    multiplier,  # scalar multiplier
    activation_type: tl.constexpr,  # activation type: 0 for sigmoid, 1 for relu
    BLOCK_SIZE: tl.constexpr  # block size for parallel execution
):
    # Compute the program ID and block indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from pointers
    x = tl.load(x_ptr + offsets, mask=offsets < x_ptr.shape[0])
    bias = tl.load(bias_ptr + offsets, mask=offsets < bias_ptr.shape[0])
    in_data = tl.load(in_ptr + offsets, mask=offsets < in_ptr.shape[0])

    # Perform fused operations: add, multiply, and activate
    x = x + bias + multiplier * in_data
    if activation_type == 0:  # Sigmoid activation
        x = 1 / (1 + tl.exp(-x))
    elif activation_type == 1:  # ReLU activation
        x = tl.max(x, 0)

    # Store the result back to x_ptr
    tl.store(x_ptr + offsets, x, mask=offsets < x_ptr.shape[0])

def fused_add_mul_activation_torch(x, bias, in_tensor, multiplier=1.0, activation='sigmoid'):
    assert x.shape == bias.shape == in_tensor.shape, "All input tensors must have the same shape"

    # Determine activation type
    activation_type = 0 if activation == 'sigmoid' else 1 if activation == 'relu' else None
    if activation_type is None:
        raise ValueError("Unsupported activation type. Use 'sigmoid' or 'relu'.")

    # Define block size and grid
    BLOCK_SIZE = 128  # Example block size
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)

    # Launch Triton kernel
    fused_add_mul_activation_kernel[grid](
        x, bias, in_tensor, multiplier, activation_type, BLOCK_SIZE
    )

    return x
