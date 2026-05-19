import triton
import triton.language as tl
from triton.language.extra import libdevice

# Asin Kernel
# This is the kernel function for calculating the asin (arc sine) of elements in the input tensor.
# The function uses libdevice.asin to perform the computation.

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # Get the program ID for the current block
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for elements in the block
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    x = tl.load(x_ptr + offsets, mask=mask)  # Load input tensor values
    
    # Check if input values are within the valid range [-1, 1]
    valid_mask = tl.bitwise_and(x >= -1.0, x <= 1.0)
    
    # Apply the asin function from libdevice only if the input value is valid
    x = tl.where(valid_mask, libdevice.asin(x), float('nan'))
    
    tl.store(y_ptr + offsets, x, mask=mask)  # Store the result in the output tensor


# Using the asin kernel
# This is the function that invokes the asin_kernel with proper parameters.

def asin(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    input = input.contiguous()
    out = out.contiguous()
    
    n_elements = input.numel()  # Get the number of elements in the tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )  # Define the grid dimensions for Triton
    
    asin_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)  # Launch the kernel
    
    return out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432  # Size of the input tensor
    x = torch.rand(size, device='cuda')  # Create a random input tensor
    output_triton = torch.empty_like(x, device='cuda')  # Output tensor for Triton results
    output_torch = torch.asin(x)  # Compute the arc sine using PyTorch

    output_triton = asin(x, output_triton)  # Compute the arc sine using the custom Triton implementation

    # Print the results for comparison
    print("PyTorch Result:")
    print(output_torch)
    print("Triton Result:")
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
