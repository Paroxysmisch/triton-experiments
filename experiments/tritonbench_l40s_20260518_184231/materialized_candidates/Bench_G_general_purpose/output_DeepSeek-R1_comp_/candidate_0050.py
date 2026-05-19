import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(input >= 0, input, tl.zeros_like(input))
    tl.store(output_ptr + offsets, output, mask=mask)

def relu(input_tensor):
    if not input_tensor.is_cuda:
        raise RuntimeError("Input tensor must be on GPU.")
    output_tensor = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    relu_kernel[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024
    )
    return output_tensor
