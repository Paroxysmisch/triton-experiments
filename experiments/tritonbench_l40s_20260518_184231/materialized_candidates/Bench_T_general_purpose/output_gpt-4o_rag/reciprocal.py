import triton
import triton.language as tl
import torch
import math

@triton.jit
def reciprocal_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Kernel function to compute the reciprocal of elements in input tensor and store them in output tensor.
    
    Parameters:
    input_ptr (tensor): Input tensor with shape (n_elements,)
    output_ptr (tensor): Output tensor to store reciprocal values
    n_elements (int): Total number of elements in the input tensor
    BLOCK_SIZE (int): Block size used for parallel computation
    """
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_val = tl.load(input_ptr + offset, mask=mask)
    reciprocal_val = 1 / input_val.to(tl.float32)
    tl.store(output_ptr + offset, reciprocal_val, mask=mask)

def reciprocal(input, *, out=None):
    """
    Wrapper function to call the Triton kernel for computing the reciprocal of elements in the input tensor.
    
    Parameters:
    input (Tensor): Input tensor
    out (Tensor, optional): Output tensor to store reciprocal values
    
    Returns:
    Tensor: Tensor with reciprocal values
    """
    # Promote integral inputs to the default scalar type (float32)
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        input = input.to(torch.get_default_dtype())
    
    # If out is not provided, create an output tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    reciprocal_kernel[(grid_size,)](input, out, n_elements, block_size)
    
    return out

# Example usage
if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    print("Input Tensor:", a)
    result = reciprocal(a)
    print("Reciprocal Tensor:", result)
