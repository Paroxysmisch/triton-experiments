import torch
import triton
import triton.language as tl

def airy_ai(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on CUDA device
    if not input.is_cuda:
        input = input.cuda()

    # If no output tensor is provided, create one
    if out is None:
        out = torch.empty_like(input)

    # Ensure the output tensor is on CUDA device
    if not out.is_cuda:
        out = out.cuda()

    # Define the Triton kernel for the Airy function Ai
    @triton.jit
    def kernel(input_ptr, out_ptr, n_elements):
        pid = tl.program_id(axis=0)
        block_start = pid * 1024
        offsets = block_start + tl.arange(0, 1024)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        # Call the C library function for the Airy function Ai
        result = libdevice.airy_ai(x)
        tl.store(out_ptr + offsets, result, mask=mask)

    # Launch the Triton kernel with a grid size of (n_elements + 1023) // 1024
    grid = lambda meta: (triton.cdiv(input.numel(), 1024),)
    kernel[grid](input, out, input.numel())
    return out
