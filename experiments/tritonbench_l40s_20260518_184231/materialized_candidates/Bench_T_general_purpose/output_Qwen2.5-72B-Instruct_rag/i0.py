import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def i0_kernel(
    input_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Initialize the output with 1.0 (the first term of the series)
    output_val = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)

    # Compute the series sum
    k = 0
    while True:
        term = (input_val * input_val / 4.0) ** k / (tl.math.factorial(k) * tl.math.factorial(k))
        output_val += term
        k += 1
        if tl.all(term < 1e-6, axis=0):
            break

    tl.store(output_ptr + offsets, output_val, mask=mask)

def i0(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    input = input.to(device)
    
    # Determine the size of the input tensor
    size = input.numel()
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input tensor shape"
        assert out.is_cuda, "Output tensor must be on the same device as the input tensor"
    
    # Determine the block size for parallel execution
    BLOCK_SIZE = triton.next_power_of_2(size)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    i0_kernel[grid](
        input_ptr=input, output_ptr=out, size=size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
