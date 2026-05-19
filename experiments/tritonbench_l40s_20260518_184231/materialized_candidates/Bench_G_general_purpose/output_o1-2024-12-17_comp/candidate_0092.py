import triton
import triton.language as tl
import torch

@triton.jit
def puzzle1_kernel(
    x_ptr,                # Pointer to input tensor
    output_ptr,           # Pointer to output tensor
    N,                    # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size
    value                 # Constant value to add
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_data = tl.load(x_ptr + offsets, mask=mask)
    x_data = x_data + value
    tl.store(output_ptr + offsets, x_data, mask=mask)

def puzzle1(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda, "Input and output must be on GPU."
    N = x.numel()

    def grid(meta):
        return ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    return output
