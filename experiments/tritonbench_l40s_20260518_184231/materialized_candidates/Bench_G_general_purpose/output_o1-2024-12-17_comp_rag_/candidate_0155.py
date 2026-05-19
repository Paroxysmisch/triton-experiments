import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,              # Pointer to input/output tensor
    in_ptr,             # Pointer to tensor to be scaled and added
    bias_ptr,           # Pointer to bias tensor
    n_elements,         # Total number of elements
    scale,              # Scaling factor
    ACT: tl.constexpr,  # Activation type (0=none, 1=sigmoid, 2=relu)
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last")
    in_val = tl.load(in_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last")
    bias_val = tl.load(bias_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last")

    out_val = x_val + scale * in_val + bias_val

    if ACT == 1:  # Sigmoid
        out_val = 1.0 / (1.0 + tl.exp(-out_val))
    elif ACT == 2:  # ReLU
        out_val = tl.maximum(out_val, 0.0)

    tl.store(x_ptr + offsets, out_val, mask=mask, eviction_policy="evict_last")


def fused_add_mul_activation_torch(
    x: torch.Tensor,
    in_tensor: torch.Tensor,
    bias: torch.Tensor,
    scale: float = 1.0,
    activation: str = "sigmoid"
):
    assert x.is_cuda and in_tensor.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors."
    assert x.is_contiguous() and in_tensor.is_contiguous() and bias.is_contiguous(), "All tensors must be contiguous."
    
    n_elements = x.numel()
    assert in_tensor.numel() == n_elements, "Input tensor sizes must match."
    assert bias.numel() == n_elements, "Bias tensor sizes must match."

    # Determine activation code
    act_map = {"none": 0, "sigmoid": 1, "relu": 2}
    ACT = act_map.get(activation, 0)

    # Set block size
    BLOCK_SIZE = 1024

    # Define grid
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_add_mul_activation_kernel[grid](
        x, 
        in_tensor,
        bias,
        n_elements,
        scale,
        ACT,
        BLOCK_SIZE
    )

    return x
