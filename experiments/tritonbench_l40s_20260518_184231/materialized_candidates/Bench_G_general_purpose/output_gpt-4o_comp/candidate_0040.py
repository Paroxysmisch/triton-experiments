import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr, p_ptr, p_mask_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index
    block_idx = tl.program_id(0)
    # Calculate the start index for this block
    start_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Check bounds
    mask = start_idx < n_elements

    # Load elements from p_mask_ptr and convert them to boolean masks
    p_mask = tl.load(p_mask_ptr + start_idx, mask=mask).to(tl.int1)

    # Load elements from grad_ptr and p_ptr
    grad = tl.load(grad_ptr + start_idx, mask=mask)
    p = tl.load(p_ptr + start_idx, mask=mask)

    # Perform the masked addition
    result = grad + p * alpha * p_mask

    # Store the result back to grad_ptr
    tl.store(grad_ptr + start_idx, result, mask=mask)


def masked_add(grad, p, p_mask, alpha, BLOCK_SIZE=1024):
    # Ensure all tensors are on CUDA
    assert grad.is_cuda and p.is_cuda and p_mask.is_cuda
    # Ensure all tensors have the same layout and strides
    assert grad.stride() == p.stride() == p_mask.stride()
    assert grad.size() == p.size() == p_mask.size()

    # Determine the number of elements
    n_elements = grad.numel()

    # Calculate the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_add_kernel[grid_size](
        grad.data_ptr(), p.data_ptr(), p_mask.data_ptr(), alpha, n_elements, BLOCK_SIZE
    )
