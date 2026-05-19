import triton
import triton.language as tl
import torch

@triton.jit
def masked_select_kernel(
    inp_ptr,                                    # pointer to input tensor
    select_mask_ptr,                            # pointer to mask tensor
    prefix_sum_ptr,                             # pointer to prefix sum array
    out_ptr,                                    # pointer to output tensor
    n_elements,                                 # total number of elements
    BLOCK_SIZE: tl.constexpr                    # size of thread block
):
    # Calculate global thread ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input and mask values
    x = tl.load(inp_ptr + offsets, mask=mask)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask)
    prefix_sum = tl.load(prefix_sum_ptr + offsets, mask=mask)
    
    # Convert mask to boolean
    select_mask = select_mask.to(tl.int1)
    
    # Store selected elements to output
    output_mask = mask & select_mask
    tl.store(out_ptr + prefix_sum - 1, x, mask=output_mask)

def masked_select(input_tensor, mask):
    # Ensure input and mask can be broadcast
    if not broadcastable(input_tensor.shape, mask.shape):
        raise ValueError("Input and mask must be broadcastable")
    
    # Broadcast tensors if needed
    if input_tensor.shape != mask.shape:
        input_tensor = input_tensor.expand_as(mask)
    
    # Flatten tensors
    input_flat = input_tensor.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1)
    
    # Calculate prefix sum for output positions
    prefix_sum = torch.cumsum(mask_flat.int(), dim=0)
    output_size = int(prefix_sum[-1].item())
    
    # Prepare output tensor
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    n_elements = input_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    masked_select_kernel[grid](
        input_flat.data_ptr(),
        mask_flat.data_ptr(),
        prefix_sum.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024
    )
    
    return output

def broadcastable(shape1, shape2):
    """Check if two shapes are broadcastable."""
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)
    shape1 = (1,) * (max_len - len1) + shape1
    shape2 = (1,) * (max_len - len2) + shape2
    
    return all((s1 == s2) or (s1 == 1) or (s2 == 1)
               for s1, s2 in zip(shape1, shape2))

# Configuration auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements']
)
