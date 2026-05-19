# triton_kernel.py
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements):
    # Compute the reciprocal of the square root
    idx = tl.program_id(0)
    if idx < n_elements:
        value = tl.load(input_ptr + idx)
        # Handle negative values
        result = tl.where(value < 0, tl.nan, 1.0 / tl.sqrt(value))
        tl.store(output_ptr + idx, result)

def rsqrt(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the kernel
    n_elements = input.numel()
    grid = (n_elements,)
    rsqrt_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
