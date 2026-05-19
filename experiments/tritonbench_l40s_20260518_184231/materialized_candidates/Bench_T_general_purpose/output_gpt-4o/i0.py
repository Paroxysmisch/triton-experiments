import triton
import triton.language as tl
import torch

# Triton kernel for computing the zeroth order modified Bessel function of the first kind
@triton.jit
def bessel_i0_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Define a range for the block
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    # Initialize the output elements
    output_elements = tl.zeros_like(input_elements)
    
    # Calculate the Bessel function using the series expansion
    k = 0
    term = (input_elements ** 2) / 4.0
    current_term = tl.ones_like(input_elements)
    while k < 100:  # Use a fixed number of iterations for convergence
        output_elements += current_term
        k += 1
        current_term *= term / (k * k)
        # Break the loop if terms become too small to contribute
        if tl.all(current_term < 1e-10):
            break

    # Store the result
    tl.store(output_ptr + offsets, output_elements, mask=offsets < n_elements)

# Wrapper function
def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous() and out.is_contiguous(), "Input and output tensors must be contiguous"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # Define a block size

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bessel_i0_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage
input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
output_tensor = i0(input_tensor)
print(output_tensor)
