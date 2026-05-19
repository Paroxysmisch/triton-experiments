import torch
import triton
import triton.language as tl

def broadcastable(shape_a, shape_b):
    for a, b in zip(shape_a[::-1], shape_b[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def masked_select_kernel(
    inp_ptr,
    mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = pid < n_elements
    
    # Load data and mask values
    mask_val = tl.load(mask_ptr + pid, mask=active, other=0)
    input_val = tl.load(inp_ptr + pid, mask=active)
    prefix = tl.load(prefix_sum_ptr + pid, mask=active) - 1
    
    # Convert mask to boolean
    cond = mask_val.to(tl.int1, bitcast=True)
    write_mask = active & cond
    
    # Scatter selected values
    tl.store(out_ptr + prefix, input_val, mask=write_mask)

def masked_select(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and mask.is_cuda, "Tensors must be on GPU"
    assert mask.dtype == torch.bool, "Mask must be boolean"
    assert broadcastable(x.shape, mask.shape), "Incompatible shapes"

    # Broadcast and flatten tensors
    x, mask = torch.broadcast_tensors(x, mask)
    x_flat = x.reshape(-1)
    mask_flat = mask.reshape(-1)
    n_elements = x_flat.numel()
    
    if n_elements == 0:
        return torch.empty_like(x_flat[:0])
    
    # Compute prefix sum
    mask_int = mask_flat.to(torch.int32)
    prefix_sum = torch.cumsum(mask_int, dim=0)
    output_size = prefix_sum[-1].item() if n_elements > 0 else 0
    
    if output_size == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)
    
    # Allocate output tensor
    output = torch.empty(output_size, dtype=x.dtype, device=x.device)

    # Launch kernel with dynamic grid
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    masked_select_kernel[grid](x_flat, mask_int, prefix_sum, output, n_elements)
    
    return output

x = torch.rand(1000, 1000, device='cuda')
mask = torch.rand(1000, 1000, device='cuda') > 0.5
result = masked_select(x, mask)
