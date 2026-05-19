import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Compute block start and end
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Compute offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input and mask
    x = tl.load(inp_ptr + offsets, mask=mask)
    m = tl.load(mask_ptr + offsets, mask=mask)
    
    # Compute prefix sum of mask
    prefix_sum = tl.cumsum(m)
    
    # Compute output offsets
    output_offsets = prefix_sum - 1
    
    # Store selected elements to output
    tl.store(out_ptr + output_offsets, x, mask=m)

# Wrapper function
def masked_select(inp, mask):
    assert inp.shape == mask.shape, "Input and mask shapes must match"
    n_elements = inp.numel()
    
    # Determine output size
    out_numel = mask.sum().item()
    output = torch.empty(out_numel, dtype=inp.dtype, device=inp.device)
    
    # Configure kernel
    def cfggen():
        for block_size in [128, 256, 512, 1024]:
            for num_warps in [4, 8]:
                yield triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
    
    # Launch kernel
    masked_select_kernel[cfggen()](
        inp.data_ptr(),
        mask.data_ptr(),
        output.data_ptr(),
        n_elements,
    )
    
    return output

# Helper function to check if shapes are broadcastable
def broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True
