import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(input_ptr, index_ptr, mask_ptr, out_ptr, dim, value, num_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index of the element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements

    # Load index and mask
    indices = tl.load(index_ptr + idx, mask=mask, other=0)
    masks = tl.load(mask_ptr + idx, mask=mask, other=False)

    # Compute the offset for the gather operation
    input_offsets = idx * dim + indices
    gathered_values = tl.load(input_ptr + input_offsets, mask=mask, other=0)

    # Apply the masked fill
    result = tl.where(masks, value, gathered_values)

    # Store the result
    tl.store(out_ptr + idx, result, mask=mask)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    assert input.dim() == index.dim(), "Input and index tensors must have the same number of dimensions."
    assert mask.dim() == input.dim(), "Mask tensor must be broadcastable to the shape of the gathered output."

    # Flatten input, index, and mask for 1D processing
    input_flat = input.flatten()
    index_flat = index.flatten()
    mask_flat = mask.flatten()

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(index_flat, dtype=input.dtype)

    # Calculate number of elements
    num_elements = index_flat.numel()

    # Define block size
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU capabilities

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    fused_gather_masked_fill_kernel[grid](input_flat, index_flat, mask_flat, out, dim, value, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape the output to the original index shape
    return out.view(index.shape)

# Example usage
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
index_tensor = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
mask_tensor = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
value = 10.0

output_tensor = fused_gather_masked_fill(input_tensor, 1, index_tensor, mask_tensor, value)
print(output_tensor)
