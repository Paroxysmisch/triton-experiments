import triton
import triton.language as tl
import torch

# Triton kernel for masked select operation
@triton.jit
def masked_select_kernel(inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the block and thread indices
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Check bounds
    mask = offsets < n_elements

    # Load input and mask values
    inp = tl.load(inp_ptr + offsets, mask=mask, other=0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)

    # Load prefix sum values
    prefix_sum = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0)

    # Calculate output positions
    output_positions = prefix_sum - 1

    # Store selected values to output buffer
    tl.store(out_ptr + output_positions, inp, mask=select_mask & mask)

# Utility function to check if two shapes are broadcastable
def broadcastable(shape1, shape2):
    for s1, s2 in zip(reversed(shape1), reversed(shape2)):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True

# Wrapper function for masked select operation
def masked_select(inp, select_mask):
    # Ensure inputs are compatible for broadcasting
    if not broadcastable(inp.shape, select_mask.shape):
        raise ValueError("Input and mask shapes are not broadcastable")

    # Flatten input and mask
    inp_flat = inp.flatten()
    select_mask_flat = select_mask.flatten()

    # Compute prefix sum of the mask
    prefix_sum = torch.cumsum(select_mask_flat, dim=0)

    # Allocate output buffer
    out_size = prefix_sum[-1].item()
    out = torch.empty(out_size, dtype=inp.dtype, device=inp.device)

    # Define block size
    BLOCK_SIZE = 1024

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(inp_flat.numel(), meta['BLOCK_SIZE']),)
    masked_select_kernel[grid](inp_flat, select_mask_flat, prefix_sum, out, inp_flat.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return out

# Configuration generator for autotuning
def cfggen():
    configs = []
    for block_size in [64, 128, 256, 512, 1024]:
        for num_warps in [1, 2, 4, 8]:
            configs.append(triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps))
    return configs

# Example usage
inp = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
select_mask = torch.tensor([0, 1, 0, 1, 1], dtype=torch.int32, device='cuda')
output = masked_select(inp, select_mask)
print(output)  # Should print tensor([2., 4., 5.], device='cuda:0')
