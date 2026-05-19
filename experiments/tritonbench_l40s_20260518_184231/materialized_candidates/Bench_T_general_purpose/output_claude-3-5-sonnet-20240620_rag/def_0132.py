import triton
import triton.language as tl
import torch

# Kernel for element-wise multiplication and subtraction
@triton.jit
def mul_sub_kernel(input_ptr, other_mul_ptr, other_sub_ptr, out_ptr, n_elements, alpha: tl.constexpr):
    """
    A kernel to perform element-wise multiplication and subtraction.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul_ptr: Pointer to the tensor or scalar for multiplication.
    - other_sub_ptr: Pointer to the tensor or scalar for subtraction.
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensors.
    - alpha: Scalar multiplier for the subtraction.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * tl.numel()
    offsets = block_start + tl.arange(0, tl.numel())
    mask = offsets < n_elements

    input_tensor = tl.load(input_ptr + offsets, mask=mask)
    other_mul = tl.load(other_mul_ptr + offsets, mask=mask) if other_mul_ptr is not None else 1
    other_sub = tl.load(other_sub_ptr + offsets, mask=mask) if other_sub_ptr is not None else 0

    out = (input_tensor * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, out, mask=mask)

# Wrapper function for the multiplication and subtraction operation
def mul_sub(input: torch.Tensor, other_mul: torch.Tensor or float, other_sub: torch.Tensor or float, alpha: float = 1, out: torch.Tensor = None) -> torch.Tensor:
    """
    A wrapper function to perform element-wise multiplication and subtraction.
    
    Parameters:
    - input: The input tensor to be multiplied.
    - other_mul: The tensor or number to multiply with `input`.
    - other_sub: The tensor or number to subtract from the multiplication result.
    - alpha: The multiplier for `other_sub`. Default is 1.
    - out: The output tensor (optional).
    
    Returns:
    - The resulting tensor after performing the operation.
    """
    n_elements = input.numel()
    if out is None:
        out = torch.empty_like(input)

    block_size = triton.next_power_of_2(n_elements)
    grid_size = triton.cdiv(n_elements, block_size)

    # Prepare pointers for the kernel
    other_mul_ptr = other_mul if isinstance(other_mul, torch.Tensor) else None
    other_sub_ptr = other_sub if isinstance(other_sub, torch.Tensor) else None

    # Call the kernel
    mul_sub_kernel[(grid_size, 1, 1)](input, other_mul_ptr, other_sub_ptr, out, n_elements, alpha)
    
    return out
