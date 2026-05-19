import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def i0_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID determines which block of data this kernel is working on
    pid = tl.program_id(axis=0)
    
    # Offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < n_elements
    
    # Load input values with masking
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize the output value
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute the Bessel function I0 using a series expansion
    term = tl.ones([BLOCK_SIZE], dtype=tl.float32)
    x2 = x * x / 4.0
    k = 0
    while tl.any(term > 1e-10):
        result += term
        k += 1
        term *= x2 / (k * k)
    
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, result, mask=mask)


def i0(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure input is on the correct device
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    # Number of elements in the input tensor
    n_elements = input.numel()
    
    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    i0_kernel[grid_size](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Example usage
input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
output_tensor = i0(input_tensor)
print(output_tensor)
