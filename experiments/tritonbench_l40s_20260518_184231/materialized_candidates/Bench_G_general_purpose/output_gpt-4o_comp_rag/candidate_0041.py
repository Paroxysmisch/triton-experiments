import triton
import triton.language as tl
import torch

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    scalar,  # Scalar value
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size for tiling
):
    # Calculate the block and thread index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boundary check to ensure safe memory operations
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute the power operation
    y = tl.pow(x, scalar)

    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    x: torch.Tensor,  # Input tensor
    scalar: float,  # Scalar value
    output: torch.Tensor = None  # Output tensor (optional)
):
    # Ensure the input tensor is contiguous
    x = x.contiguous()

    # Allocate output tensor if not provided
    if output is None:
        output = torch.empty_like(x)

    # Get the number of elements in the tensor
    n_elements = x.numel()

    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        x_ptr=x.data_ptr(),
        y_ptr=output.data_ptr(),
        scalar=scalar,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda')
scalar = 2.0
output = pow_func_scalar_tensor_wrapper_rank_1(x, scalar)
print(output)
