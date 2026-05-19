import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,  # Pointer to the grad tensor
    p_data_ptr,  # Pointer to the p_data tensor
    mask_ptr,  # Pointer to the mask tensor
    grad_stride,  # Stride of the grad tensor
    p_data_stride,  # Stride of the p_data tensor
    mask_stride,  # Stride of the mask tensor
    alpha,  # Scaling factor
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the data
    grad = tl.load(grad_ptr + offsets, mask=offsets < n_elements)
    p_data = tl.load(p_data_ptr + offsets, mask=offsets < n_elements)
    mask = tl.load(mask_ptr + offsets, mask=offsets < n_elements)

    # Apply the mask and add
    result = tl.where(mask == 0, grad + p_data * alpha, grad)

    # Store the result back to grad
    tl.store(grad_ptr + offsets, result, mask=offsets < n_elements)

import torch

def masked_add(grad, p_data, mask, alpha):
    assert grad.shape == p_data.shape == mask.shape, "Input tensors must have the same shape"
    assert grad.is_cuda and p_data.is_cuda and mask.is_cuda, "Input tensors must be on the same GPU"

    n_elements = grad.numel()
    grid = (triton.cdiv(n_elements, 1024),)

    grad_ptr = grad.data_ptr()
    p_data_ptr = p_data.data_ptr()
    mask_ptr = mask.data_ptr()

    grad_stride = grad.stride(0)
    p_data_stride = p_data.stride(0)
    mask_stride = mask.stride(0)

    masked_add_kernel[grid](
        grad_ptr,
        p_data_ptr,
        mask_ptr,
        grad_stride,
        p_data_stride,
        mask_stride,
        alpha,
        n_elements,
        BLOCK_SIZE=1024
    )

# Example usage
if __name__ == "__main__":
    grad = torch.randn(1024, device='cuda')
    p_data = torch.randn(1024, device='cuda')
    mask = torch.randint(0, 2, (1024,), device='cuda', dtype=torch.int32)
    alpha = 0.5

    masked_add(grad, p_data, mask, alpha)

    print("grad after masked_add:", grad)
