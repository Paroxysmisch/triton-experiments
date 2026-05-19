import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr, mask_ptr, out_ptr, n_elements, out_numel,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the offsets for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (within n_elements)
    mask = offsets < n_elements
    
    # Load input data and selection mask
    x = tl.load(inp_ptr + offsets, mask=mask)
    m = tl.load(mask_ptr + offsets, mask=mask)
    
    # Compute prefix sum of the mask to get output offsets
    prefix_sum = tl.cumsum(m)
    output_offset = tl.sum(m) - 1
    
    # Store selected elements to output
    tl.store(out_ptr + output_offset, x, mask=m & mask)

# Wrapper function
def masked_select(inp, mask):
    assert inp.shape == mask.shape, "Input and mask shapes must match"
    n_elements = inp.numel()
    out_numel = mask.sum().item()
    
    # Allocate output tensor
    out = torch.empty(out_numel, dtype=inp.dtype, device=inp.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_select_kernel[grid](inp, mask, out, n_elements, out_numel, BLOCK_SIZE=1024)
    
    return out

# Configuration generator
def cfggen():
    return [
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ]

# Broadcastable function
def broadcastable(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True
