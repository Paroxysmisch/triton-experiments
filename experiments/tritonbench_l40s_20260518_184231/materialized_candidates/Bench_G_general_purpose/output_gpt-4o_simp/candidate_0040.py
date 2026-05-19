import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr, p_data_ptr, mask_ptr, alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(axis=0)

    # Block indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from global memory
    grad = tl.load(grad_ptr + offsets, mask=offsets < n_elements, other=0.0)
    p_data = tl.load(p_data_ptr + offsets, mask=offsets < n_elements, other=0.0)
    mask = tl.load(mask_ptr + offsets, mask=offsets < n_elements, other=1.0)

    # Apply the mask: update grad only where mask is 0
    update = tl.where(mask == 0, p_data * alpha, 0.0)
    grad += update

    # Store the result back to global memory
    tl.store(grad_ptr + offsets, grad, mask=offsets < n_elements)

def masked_add(grad, p_data, mask, alpha, block_size=1024):
    # Ensure the input tensors are contiguous
    assert grad.is_contiguous()
    assert p_data.is_contiguous()
    assert mask.is_contiguous()

    # Number of elements
    n_elements = grad.numel()

    # Launch the kernel
    grid = (triton.cdiv(n_elements, block_size),)
    masked_add_kernel[grid](
        grad, p_data, mask, alpha,
        n_elements,
        BLOCK_SIZE=block_size
    )
