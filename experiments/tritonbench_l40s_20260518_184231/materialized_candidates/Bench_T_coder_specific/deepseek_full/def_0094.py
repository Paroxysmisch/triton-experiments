import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def dropout_linear_relu_kernel(
    input_ptr, weight_ptr, dropout_ptr, output_ptr,
    n_rows, n_cols,
    in_feat_stride, out_feat_stride, dropout_stride,
    dropout_rate, training,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    row_end = row_start + BLOCK_SIZE
    row_mask = (row_start < n_rows) & (row_end <= n_rows)

    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_rows, n_cols),
        strides=(in_feat_stride, 1),
        offsets=(row_start, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )

    cols = tl.arange(0, BLOCK_SIZE)
    weight_ptrs = weight_ptr + cols * out_feat_stride
    out = tl.load(block_ptr, boundary_check=(0, 1))
    out = tl.dot(out, tl.trans(tl.load(weight_ptrs, boundary_check=(0,))), allow_tf32=False)

    if bias is not None:
        bias_ptrs = bias_ptr + cols
        bias = tl.load(bias_ptrs, mask=cols < BLOCK_SIZE, other=0.0)
        out = out + bias

    out = tanh(out)

    if training:
        keep_mask = tl.rand(tl.load(block_ptr, boundary_check=(0, 1)).shape, dtype=tl.float32) > dropout_rate
        keep_mask = tl.trans(keep_mask)
        out = tl.where(keep_mask, out / (1.0 - dropout_rate), 0.0)
        tl.store(dropout_ptr + row_start * dropout_stride, keep_mask, boundary_check=(0,))

    out_ptr = output_ptr + row_start * out_feat_stride
    tl.store(out_ptr, out, boundary_check=(0,))

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert (bias is None) or bias.is_contiguous()
    assert 0.0 <= p <= 1.0

    n_rows, n_cols = input.shape
    n_outs = weight.shape[0]

    output = torch.empty((n_rows, n_outs), device=input.device, dtype=torch.float32) if not inplace else input
    if training:
        dropout_rate = 0.5 * (1.0 + p)
        dropout_mask = torch.empty((n_rows, n_cols), device=input.device, dtype=torch.bool)
    else:
        dropout_rate = 0.0
        dropout_mask = None

    grid = lambda meta: (triton.cdiv(n_rows, meta['BLOCK_SIZE']),)

    dropout_linear_relu_kernel[grid](
        input, weight, dropout_mask, output,
        n_rows, n_cols,
        input.stride(0), weight.stride(0), dropout_mask.stride(0),
        dropout_rate, training,
    )

    if dropout_mask is not None:
        output = torch.where(dropout_mask, output / (1.0 - dropout_rate), 0.0)

    return output
