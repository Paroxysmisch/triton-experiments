import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr,  # *Pointer* to first input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   # NOTE: `constexpr` so it can be used as a shape value.
                   ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    output = 1 / (1 + tl.exp(-x))

    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output
