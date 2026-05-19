import torch
import triton
import triton.language as tl

# Kernel function
@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    tensor_ptr, scalar, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block and thread index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boundary check
    mask = offsets < n_elements

    # Load the input data
    tensor_data = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)

    # Compute the power
    result = tl.pow(tensor_data, scalar)

    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

# Wrapper function
def pow_func_scalar_tensor_wrapper_rank_1(tensor, scalar):
    # Ensure input is a torch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Get the number of elements
    n_elements = tensor.numel()

    # Allocate output tensor
    output = torch.empty_like(tensor)

    # Define the block size and number of blocks
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU's capabilities
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        tensor_ptr=tensor,
        scalar=scalar,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
scalar = 2.0
result = pow_func_scalar_tensor_wrapper_rank_1(tensor, scalar)
print(result)  # Output should be [1.0, 4.0, 9.0, 16.0]
