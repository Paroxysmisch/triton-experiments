import triton
import triton.language as tl
import torch
import math

@triton.jit
def reciprocal_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Kernel function to compute the reciprocal of elements in array 'a' and store them in 'b'.
    
    Parameters:
    a (tensor): Input tensor with shape (n_elements,)
    b (tensor): Output tensor to store reciprocal values of 'a'
    n_elements (int): Total number of elements in the input tensor 'a'
    BLOCK_SIZE (int): Block size used for parallel computation
    """
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = 1 / a_value.to(tl.float32)  # Promote to default scalar type
    tl.store(b + offset, b_value, mask=mask)  

def reciprocal(input, *, out=None):
    """
    Wrapper function to call the Triton kernel for computing the reciprocal of elements in 'input'.
    
    Parameters:
    input (tensor): Input tensor with shape (n_elements,)
    out (tensor, optional): Output tensor to store reciprocal values. If None, a new tensor is created.
    
    Returns:
    tensor: Output tensor with reciprocal values
    """
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    reciprocal_func[(grid_size, 1, 1)](input, out, n_elements, block_size)
    return out

if __name__ == "__main__":
    a = torch.randn([4, 4], device="cuda")
    print(torch.reciprocal(a))
    print(reciprocal(a))
