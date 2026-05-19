import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,  # Input tensor pointer
    in_ptr,  # Input tensor pointer to be scaled and added
    bias_ptr,  # Bias tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    scale,  # Scaling factor for in_ptr
    ACTIVATION: tl.constexpr,  # Activation type (0 for sigmoid, 1 for relu)
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for the block
    mask = offsets < n_elements  # Create a mask to handle the last block

    x = tl.load(x_ptr + offsets, mask=mask)  # Load x values
    in_val = tl.load(in_ptr + offsets, mask=mask)  # Load in values
    bias = tl.load(bias_ptr + offsets, mask=mask)  # Load bias values

    # Perform the fused operation
    result = x + bias + in_val * scale

    # Apply the activation function
    if ACTIVATION == 0:
        result = 1 / (1 + tl.exp(-result))  # Sigmoid activation
    elif ACTIVATION == 1:
        result = tl.where(result > 0, result, 0)  # ReLU activation

    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, result, mask=mask)

import torch

def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    in_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    scale: float = 1.0,
    activation: str = 'sigmoid',
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
):
    # Ensure the tensors are on the same device
    assert in_out_tensor.device == in_tensor.device == bias_tensor.device, "All tensors must be on the same device"
    assert in_out_tensor.dtype == in_tensor.dtype == bias_tensor.dtype, "All tensors must have the same data type"

    # Determine the number of elements
    n_elements = in_out_tensor.numel()

    # Set the default grid size
    if max_grid is None:
        max_grid = (8192, 1, 1)

    # Map activation string to activation code
    activation_code = 0 if activation == 'sigmoid' else 1 if activation == 'relu' else None
    assert activation_code is not None, "Unsupported activation function. Use 'sigmoid' or 'relu'."

    # Configure the grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid](
        in_out_tensor, in_tensor, bias_tensor, in_out_tensor,
        n_elements, scale, activation_code, BLOCK_SIZE=1024
    )

    return in_out_tensor
