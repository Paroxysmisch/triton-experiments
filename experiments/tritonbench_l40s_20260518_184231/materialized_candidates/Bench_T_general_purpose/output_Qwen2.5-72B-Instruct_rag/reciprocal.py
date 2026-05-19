import triton
import triton.language as tl
import torch

# Triton kernel to compute the reciprocal of elements in an input tensor
@triton.jit
def reciprocal_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Kernel function to compute the reciprocal of elements in array 'a' and store them in 'b'.
    
    Parameters:
    a_ptr (pointer): Pointer to the input tensor with shape (n_elements,)
    b_ptr (pointer): Pointer to the output tensor to store reciprocal values of 'a'
    n_elements (int): Total number of elements in the input tensor 'a'
    BLOCK_SIZE (int): Block size used for parallel computation
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_value = tl.load(a_ptr + offsets, mask=mask)
    b_value = 1 / a_value.to(tl.float32)
    tl.store(b_ptr + offsets, b_value, mask=mask)

# Wrapper function to call the Triton kernel
def reciprocal(input, *, out=None):
    """
    Wrapper function to compute the reciprocal of elements in the input tensor.
    
    Parameters:
    input (Tensor): The input tensor.
    out (Tensor, optional): The output tensor. If not provided, a new tensor will be created.
    
    Returns:
    Tensor: A new tensor with the reciprocal of the elements of the input tensor.
    """
    # Promote integral inputs to the default scalar type
    if input.dtype in [torch.int32, torch.int64]:
        input = input.to(torch.get_default_dtype())
    
    n_elements = input.numel()
    block_size = 1024  # Block size for parallel computation
    grid_size = triton.cdiv(n_elements, block_size)
    
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    reciprocal_kernel[(grid_size,)](input, out, n_elements, block_size)
    
    return out

# Example usage
if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    print("Input Tensor:", a)
    result = reciprocal(a)
    print("Reciprocal Tensor:", result)
