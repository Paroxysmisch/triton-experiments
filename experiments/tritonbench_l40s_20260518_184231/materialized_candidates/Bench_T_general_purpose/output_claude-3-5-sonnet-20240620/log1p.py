import triton
import triton.language as tl

@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Ensure we do not exceed the bounds
    if idx < n_elements:
        # Load the input value
        x = tl.load(input_ptr + idx)
        # Compute the natural logarithm of (1 + x)
        y = tl.log1p(x)
        # Store the result in the output tensor
        tl.store(output_ptr + idx, y)

def log1p(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Prepare the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Launch the Triton kernel
    grid = (n_elements,)
    log1p_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
