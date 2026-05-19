import torch
import triton
import triton.language as tl

@triton.jit
def ones_kernel(
    output_ptr,  # Pointer to the output tensor in GPU memory
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block of threads
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    
    out = torch.empty_like(
        input,
        dtype=dtype,
        layout=input.layout if layout is None else layout,
        device=device,
        requires_grad=requires_grad,
        memory_format=memory_format
    )
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 1024
    
    with torch.cuda.device(device):
        ones_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
