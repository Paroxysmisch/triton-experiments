import torch
import triton
import triton.language as tl

@triton.jit
def add(input, other, alpha, out):
    # Triton kernel to add the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    # The kernel uses tl.load and tl.store to access and modify the input tensor.
    # It supports broadcasting to a common shape, type promotion, and accepts integer, float, and complex inputs.
    
    # Math: out_i = input_i + alpha * other_i
    
    # Get the index of the element to process
    idx = tl.program_id(axis=0)
    
    # Load the input element at the calculated index, promoting to a common type if necessary
    input_element = tl.load(input + idx, mask=input + idx < tl.num_programs(input)).to(other.dtype.element_ty)
    
    # Load the other element, supporting broadcasting and promoting to a common type
    other_element = tl.load(other + idx % tl.num_programs(other), mask=other + idx % tl.num_programs(other) < tl.num_programs(other)).to(input.dtype.element_ty)
    
    # Perform the addition and scaling operation
    result = input_element + alpha * other_element
    
    # Store the result back into the output tensor
    tl.store(out + idx, result, mask=out + idx < tl.num_programs(out))

def add_wrapper(input, other, *, alpha=1, out=None):
    # Wrapper function for the Triton kernel 'add'.
    # If 'out' is not provided, a tensor of the same shape and type as 'input' is created.
    # The kernel is launched with the number of programs equal to the total number of elements in 'input'.
    # The function then returns the result tensor.
    
    if out is None:
        out = torch.empty_like(input)
    
    assert input.is_contiguous()
    assert (not other.is_contiguous()) or other.ndim == 1
    
    input = input.view(-1)
    other = other.view(-1) if other.ndim > 0 else other
    
    M, N = input.numel(), other.numel()
    assert other.is_contiguous()
    assert N % M == 0
    n_chunks = N // M
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add[grid](input, other, alpha, out)
    
    return out
