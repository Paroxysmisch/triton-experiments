import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,
    in_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    in_ = tl.load(in_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    x = x + bias + scale * in_
    if ACTIVATION == "sigmoid":
        x = tl.sigmoid(x)
    elif ACTIVATION == "relu":
        x = tl.relu(x)
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    in_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    scale: float = 1.0,
    activation: str = "sigmoid",
):
    assert in_out_tensor.is_contiguous()
    assert in_tensor.is_contiguous()
    assert bias_tensor.is_contiguous()
    assert in_out_tensor.ndim == 1
    assert in_tensor.ndim == 1
    assert bias_tensor.ndim == 1
    assert in_out_tensor.size(0) == bias_tensor.size(0) == in_tensor.size(0)

    n_elements = in_out_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_add_mul_activation_kernel[grid](
        in_out_tensor,
        in_tensor,
        bias_tensor,
        in_out_tensor,
        n_elements,
        scale,
        ACTIVATION=activation,
    )
    return in_out_tensor
