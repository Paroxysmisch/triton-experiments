import torch
import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_triton(
    x_ptr, weight_ptr, bias_ptr, out_ptr, x_shape, out_shape, n_features,
    p, training, inplace, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_shape[0]

    x_field_ptr = x_ptr + (offsets[:, None] * n_features + tl.arange(0, n_features)[None, :])
    x = tl.load(x_field_ptr, mask=mask, other=0.0)

    weight_field_ptr = weight_ptr + (tl.arange(0, n_features)[:, None] * out_shape[0] + offsets[None, :])
    weight = tl.load(weight_field_ptr, mask=mask.T, other=0.0)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    logits = tl.sum(x * weight, axis=1)
    if bias_ptr is not None:
        logits += bias
    out = 1 / (1 + tl.exp(-logits))
    if inplace:
        tl.store(x_ptr + offsets, out, mask=mask)
    else:
        tl.store(out_ptr + offsets, out, mask=mask)

def dropout_sigmoid_linear_triton_wrapper(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, p=0.5, training=True, inplace=False
) -> torch.Tensor:
    out = torch.empty(*x.shape, device=x.device, dtype=x.dtype)
    assert x.is_contiguous()
    if not inplace:
        assert out.is_contiguous()
    n_features = x.shape[-1]
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]),)
    dropout_sigmoid_linear_triton[grid](
        x, weight, bias, out if inplace else x, x.shape, out.shape, n_features, p, training, inplace, BLOCK_SIZE=64,
    )
    return out if not inplace else x
