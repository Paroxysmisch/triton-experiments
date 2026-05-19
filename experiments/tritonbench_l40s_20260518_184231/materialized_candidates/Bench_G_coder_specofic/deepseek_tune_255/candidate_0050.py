import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    **meta,
):
    """
    Kernel for computing the ReLU operation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * meta["BLOCK_SIZE"]
    offsets = block_start + tl.arange(0, meta["BLOCK_SIZE"])
    mask = offsets < num_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = tl.where(input_values > 0, input_values, 0)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def relu(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Applies the rectified linear unit (ReLU) activation function.
    """
    output = torch.empty_like(input)
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](input, output, num_elements, BLOCK_SIZE=1024)
    return output
