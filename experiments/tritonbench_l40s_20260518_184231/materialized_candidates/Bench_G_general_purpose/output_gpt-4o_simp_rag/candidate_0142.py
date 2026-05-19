import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the program index
    pid = tl.program_id(axis=0)
    
    # Define the start index for this block
    block_start = pid * BLOCK_SIZE

    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements, other=0.0)
    z = tl.load(z_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Perform the fused operation: (x + y) * z
    result = (x + y) * z

    # Apply an activation function, e.g., ReLU
    result = tl.max(result, 0.0)

    # Store the result back to global memory
    tl.store(out_ptr + offsets, result, mask=offsets < n_elements)

def fused_add_mul_activation_torch(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and z.is_cuda, "All tensors must be CUDA tensors"
    assert x.shape == y.shape == z.shape, "All tensors must have the same shape"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Number of elements
    n_elements = x.numel()

    # Define the block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid_size](
        x, y, z, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
