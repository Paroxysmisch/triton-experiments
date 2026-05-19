import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr,  # Pointer to the input tensor
    mask_ptr,  # Pointer to the mask tensor
    out_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the input and mask tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)

    # Load input and mask
    inp = tl.load(inp_ptr + offsets, mask=offsets < n_elements)
    mask = tl.load(mask_ptr + mask_offsets, mask=mask_offsets < n_elements)

    # Compute output offsets using prefix sum
    selected_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    selected_count = tl.zeros((1,), dtype=tl.int32)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            selected_offsets[i] = selected_count
            selected_count += 1

    # Store selected elements to the output buffer
    for i in range(BLOCK_SIZE):
        if mask[i]:
            tl.store(out_ptr + selected_offsets[i], inp[i])

import torch
import triton
import triton.language as tl

def masked_select(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert inp.is_cuda and mask.is_cuda, "Input and mask tensors must be on the same CUDA device."
    assert inp.numel() == mask.numel(), "Input and mask tensors must have the same number of elements."

    n_elements = inp.numel()
    out = torch.empty(n_elements, dtype=inp.dtype, device=inp.device)

    # Define the grid and block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_select_kernel[grid](inp, mask, out, n_elements, BLOCK_SIZE=1024)

    # Trim the output to the actual number of selected elements
    selected_count = torch.sum(mask).item()
    return out[:selected_count]

def cfggen():
    configs = []
    for block_size in [128, 256, 512, 1024]:
        for num_warps in [1, 2, 4]:
            configs.append(triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps))
    return configs

def broadcastable(shape1, shape2):
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True

# Example usage
inp = torch.tensor([1, 2, 3, 4, 5], device='cuda')
mask = torch.tensor([True, False, True, False, True], device='cuda')

selected = masked_select(inp, mask)
print(selected)  # Output: tensor([1, 3, 5], device='cuda:0')
