import torch
import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(input_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # The program id is 0 for the kernel launched for the entire tensor
    pid = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 element
    stride = tl.arange(0, BLOCK_SIZE)
    # The offsets are the locations in the input tensor that the program id points to
    offsets = pid * BLOCK_SIZE + stride
    # The mask is True for elements that we actually loaded from the tensor
    mask = offsets < n_elements
    # Load the elements from DRAM
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the natural logarithm of the absolute value of the gamma function
    result = tl.math.log(tl.math.gamma(tl.math.abs(x)))
    
    # Write the result back to DRAM
    if out_ptr is not None:
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        return result

def gammaln(input, *, out=None):
    # Ensure the input tensor is on the CUDA device
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    
    if out is not None and out.shape != input.shape:
        raise ValueError("shape of out must be the same as input")
    
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    n_elements = input.numel()
    # The block size is the smallest power of two greater than the number of elements in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    gammaln_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
