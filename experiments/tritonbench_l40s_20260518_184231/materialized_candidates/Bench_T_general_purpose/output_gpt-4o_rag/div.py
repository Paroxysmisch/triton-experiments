import triton
import triton.language as tl
import torch

# Kernel function for element-wise division
@triton.jit
def div_kernel(input_ptr, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, rounding_mode: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    input_val = tl.load(input_ptr + offset, mask=mask)
    other_val = tl.load(other_ptr + offset, mask=mask)

    # Perform division
    result = input_val / other_val

    # Apply rounding mode if specified
    if rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "ceil":
        result = tl.ceil(result)
    elif rounding_mode == "trunc":
        result = tl.trunc(result)
    # Add more rounding modes as needed

    tl.store(out_ptr + offset, result, mask=mask)

# Wrapper function for the division operation
def div(input, other, *, rounding_mode=None, out=None):
    # Ensure input and other are tensors
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=torch.get_default_dtype())
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=torch.get_default_dtype())

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input).to(torch.promote_types(input.dtype, other.dtype))

    # Determine the number of elements and block size
    n_elements = input.numel()
    block_size = triton.next_power_of_2(min(1024, n_elements))
    grid_size = triton.cdiv(n_elements, block_size)

    # Launch the kernel
    div_kernel[(grid_size,)](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
        rounding_mode=rounding_mode
    )

    return out

# Example usage
input_tensor = torch.tensor([4.0, 9.0, 16.0])
other_tensor = torch.tensor([2.0, 3.0, 4.0])
result = div(input_tensor, other_tensor, rounding_mode='floor')
print(result)
