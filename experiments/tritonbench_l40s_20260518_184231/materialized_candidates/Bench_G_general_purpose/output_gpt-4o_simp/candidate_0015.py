import triton
import triton.language as tl
import torch

@triton.jit
def masked_select_kernel(inp_ptr, mask_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index and the starting offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't read out of bounds
    mask = offsets < n_elements

    # Load input and mask data
    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)

    # Compute prefix sum to determine output positions
    selected_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    prefix_sum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    prefix_sum[0] = mask_vals[0]
    for i in range(1, BLOCK_SIZE):
        prefix_sum[i] = prefix_sum[i - 1] + mask_vals[i]

    # Store selected elements
    for i in range(BLOCK_SIZE):
        if mask_vals[i]:
            selected_offsets[i] = prefix_sum[i] - 1

    out_indices = block_start + selected_offsets
    tl.store(out_ptr + out_indices, inp, mask=mask_vals)

def masked_select(inp, mask):
    assert inp.shape == mask.shape, "Input and mask must have the same shape"
    n_elements = inp.numel()

    # Allocate output buffer
    out_size = mask.sum().item()
    out = torch.empty(out_size, dtype=inp.dtype, device=inp.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_select_kernel[grid](inp, mask, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

def cfggen():
    configs = []
    for block_size in [128, 256, 512, 1024]:
        for num_warps in [1, 2, 4]:
            configs.append(triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps))
    return configs

def broadcastable(shape1, shape2):
    # Reverse shapes for easier alignment
    shape1, shape2 = shape1[::-1], shape2[::-1]
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True

# Example usage
inp = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
mask = torch.tensor([True, False, True, False, True], dtype=torch.bool, device='cuda')
out = masked_select(inp, mask)
print(out)  # Output: tensor([1., 3., 5.], device='cuda:0')
