import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(input_ptr, other_ptr, out_ptr, alpha, dim, keepdim, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Load input and other tensors
    input_val = tl.load(input_ptr + idx)
    other_val = tl.load(other_ptr + idx) * alpha
    
    # Compute the sum
    result = input_val + other_val
    
    # Store the result
    tl.store(out_ptr + idx, result)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> Tensor:
    # Ensure input and other are tensors
    input = input.to(dtype) if dtype else input
    other = other.to(dtype) if dtype else other
    
    # Determine the shape and size
    n_elements = input.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    add_mean_kernel[(n_elements,)](input, other, out, alpha, dim, keepdim, n_elements)
    
    # Compute mean along the specified dimension
    if dim is not None:
        return out.mean(dim=dim, keepdim=keepdim)
    else:
        return out.mean()
