import triton
import triton.language as tl
import torch
import math

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_output_ptr,
    sqrt_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for the current block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load the input element and convert to float32
    input_val = tl.load(input_ptr + offset, mask=mask)
    input_float = input_val.to(tl.float32)
    # Compute the complementary error function (erfc)
    erfc_val = tl.erfc(input_float)
    # Compute the square root
    sqrt_val = tl.sqrt(input_float)
    # Store the results in the respective output tensors
    tl.store(erfc_output_ptr + offset, erfc_val, mask=mask)
    tl.store(sqrt_output_ptr + offset, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input tensor is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
    # Create output tensors with float32 dtype
    erfc_out = torch.empty_like(input, dtype=torch.float32)
    sqrt_out = torch.empty_like(input, dtype=torch.float32)
    n_elements = input.numel()
    # Calculate block and grid sizes
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the kernel
    erfc_sqrt_kernel[(grid_size, 1, 1)](
        input, erfc_out, sqrt_out, n_elements, block_size
    )
    return (erfc_out, sqrt_out)
