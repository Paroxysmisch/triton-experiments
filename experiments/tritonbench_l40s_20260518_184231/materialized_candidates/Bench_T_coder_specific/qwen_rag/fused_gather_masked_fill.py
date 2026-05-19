import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(output_ptr, input_ptr, index_ptr, mask_ptr, value, dim, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = offsets < n_elements

    # Load the mask
    mask = tl.load(mask_ptr + offsets, mask=valid_mask, other=False)

    # Gather the indices
    indices = tl.load(index_ptr + offsets, mask=valid_mask, other=0)

    # Load the input values
    input_values = tl.load(input_ptr + offsets, mask=valid_mask, other=0)

    # Perform the gather operation
    gathered_values = tl.index_select(input_values, dim, indices)

    # Fill the gathered values where the mask is True
    output_values = tl.where(mask, value, gathered_values)

    # Store the result
    tl.store(output_ptr + offsets, output_values, mask=valid_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=1, num_warps=8),
    ],
    key=['n_elements', 'dim']
)
def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    assert input.dim() == index.dim(), "Input and index must have the same number of dimensions"
    assert input.size(dim) >= index.max(), "Index out of bounds"
    assert mask.shape == input.shape[:dim] + input.shape[dim+1:], "Mask must be broadcastable to the shape of the output"

    if out is None:
        out = input.new_empty(input.shape)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    fused_gather_masked_fill_kernel[grid](out, input, index, mask, value, dim, n_elements, BLOCK_SIZE=256)

    return out
