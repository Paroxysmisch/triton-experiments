import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    in_ptr,  # Pointer to the input tensor
    bias_ptr,  # Pointer to the bias tensor
    x_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensors
    alpha,  # Scaling factor
    ACTIVATION: tl.constexpr,  # Activation function (0 for sigmoid, 1 for relu)
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    # Compute the block index
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for the block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input and bias data
    in_vec = tl.load(in_ptr + offsets, mask=mask)
    bias_vec = tl.load(bias_ptr + offsets, mask=mask)

    # Perform the fused operation
    x_vec = in_vec * alpha + bias_vec

    # Apply the activation function
    if ACTIVATION == 0:  # Sigmoid
        x_vec = 1 / (1 + tl.exp(-x_vec))
    elif ACTIVATION == 1:  # ReLU
        x_vec = tl.max(x_vec, 0.0)

    # Store the result back to the output tensor
    tl.store(x_ptr + offsets, x_vec, mask=mask)

import torch
import triton
import triton.language as tl

def fused_add_mul_activation_torch(in_out_tensor, bias_tensor, alpha=1.0, activation='sigmoid'):
    # Ensure the tensors are on the same device
    assert in_out_tensor.device == bias_tensor.device, "Input and bias tensors must be on the same device"
    device = in_out_tensor.device

    # Ensure the tensors have the same shape
    assert in_out_tensor.shape == bias_tensor.shape, "Input and bias tensors must have the same shape"

    # Define the grid and block sizes
    n_elements = in_out_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Map activation string to integer
    activation_map = {'sigmoid': 0, 'relu': 1}
    activation_code = activation_map.get(activation, 0)

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid](
        in_out_tensor,  # Pointer to the input tensor
        bias_tensor,  # Pointer to the bias tensor
        in_out_tensor,  # Pointer to the output tensor (in-place)
        n_elements,  # Number of elements in the tensors
        alpha,  # Scaling factor
        activation_code,  # Activation function code
        BLOCK_SIZE  # Block size for parallel execution
    )

    return in_out_tensor
