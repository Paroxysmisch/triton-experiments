import torch
import triton
import triton.language as tl

def broadcastable(shape_a, shape_b):
    for a, b in zip(reversed(shape_a), reversed(shape_b)):
        if a != 1 and b != 1 and a != b:
            return False
    return True

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def masked_select_kernel(
    inp_ptr, mask_ptr, prefix_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load mask and check bounds
    mask = offsets < n_elements
    cond = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    
    # Load input values and prefix sums
    vals = tl.load(inp_ptr + offsets, mask=mask)
    indices = tl.load(prefix_ptr + offsets, mask=mask)
    
    # Calculate valid stores and scatter results
    store_mask = mask & cond
    tl.store(out_ptr + indices, vals, mask=store_mask)

def masked_select(input: torch.Tensor, mask: torch.Tensor):
    # Validate inputs
    assert input.dim() == mask.dim(), "Input and mask must have same dimensions"
    assert broadcastable(input.shape, mask.shape), "Shapes not broadcastable"
    
    # Broadcast and flatten tensors
    mask = mask.expand_as(input).bool()
    input_flat = input.flatten().contiguous()
    mask_flat = mask.flatten().contiguous()
    
    n_elements = input_flat.size(0)
    if n_elements == 0:
        return torch.empty_like(input_flat[:0])
    
    # Compute prefix sum for output indexing
    inclusive_scan = torch.cumsum(mask_flat.int(), 0)
    prefix_sum = torch.empty_like(inclusive_scan)
    prefix_sum[0] = 0
    if n_elements > 1:
        prefix_sum[1:] = inclusive_scan[:-1]
    
    total = inclusive_scan[-1].item()
    if total == 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)
    
    # Allocate output and launch kernel
    output = torch.empty(total, dtype=input.dtype, device=input.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    masked_select_kernel[grid](
        input_flat, mask_flat, prefix_sum, output,
        n_elements
    )
    
    return output
