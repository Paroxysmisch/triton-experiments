import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr, 
    p_ptr, 
    p_mask_ptr, 
    alpha, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)

    out = grad + alpha * p * p_mask
    tl.store(grad_ptr + offsets, out, mask=mask)


def masked_add(grad, p, p_mask, alpha):
    assert grad.is_cuda, "grad must be a CUDA tensor"
    assert p.is_cuda, "p must be a CUDA tensor"
    assert p_mask.is_cuda, "p_mask must be a CUDA tensor"
    assert grad.shape == p.shape == p_mask.shape, "Tensors must have the same shape"
    assert grad.stride() == p.stride() == p_mask.stride(), "Tensors must have the same strides"

    n_elements = grad.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    masked_add_kernel[grid](
        grad, 
        p, 
        p_mask, 
        alpha, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
