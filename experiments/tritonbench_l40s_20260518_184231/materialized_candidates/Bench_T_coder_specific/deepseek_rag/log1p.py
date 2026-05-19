import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # The block size is the next power of two greater than n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + offsets
    # Load the input into SRAM
    inputs = tl.load(input_ptrs, mask=offsets < n_elements)
    # Compute log1p
    outputs = tl.log1p(inputs)
    # Write back output to DRAM
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, outputs, mask=offsets < n_elements)

def log1p(input, out=None):
    # Check if input and output tensors are the same
    if out is None:
        out = torch.empty_like(input)
    else:
        assert input.shape == out.shape, "Input and output tensor shapes must match"

    # The block size is the smallest power of two greater than the number of elements in `input`
    BLOCK_SIZE = triton.next_power_of_2(input.numel())
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
    log1p_kernel[(input.numel(), )](
        input,
        out,
        input.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
