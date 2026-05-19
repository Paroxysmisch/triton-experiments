import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr, stride_input, stride_weight,
    stride_out, n_features: tl.constexpr, n_out: tl.constexpr,
    last_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    input_ptr += pid * stride_input
    weight_ptr += pid * stride_weight
    out_ptr += pid * stride_out

    row_idx = tl.arange(0, BLOCK_SIZE)
    col_idx = tl.arange(0, n_features)

    input = tl.load(input_ptr + col_idx, mask=col_idx < n_features, other=0.0)
    if last_dim != -1:
        input = input * (-1)

    weight = tl.load(weight_ptr + n_features * row_idx + col_idx,
                     mask=col_idx < n_features, other=0.0)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + row_idx)
        input += bias

    linear = tl.sum(input * weight, axis=0)
    z = linear - tl.max(linear, axis=0)
    numerator = tl.exp(z)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator
    if last_dim != -1:
        softmax = tl.math.log(softmax)
    else:
        softmax = -tl.math.log(softmax)
    tl.store(out_ptr + row_idx, softmax, mask=row_idx < n_out)


def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    check_dim(dim)
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if dtype is None:
        dtype = input.dtype
    check_dtype(dtype, [torch.float16, torch.bfloat16, torch.float32])
    check_device(input.device, [torch.cuda])
    check_broadcasting(input, weight, bias)
    out = torch.empty(input_shape_broadcasted(
        input, weight, bias), dtype=dtype, device=input.device)
    n_features = input.shape[-1]
    out_features = weight.shape[0]
    assert_single_feature(input)
    last_dim = dim % input.ndim
    stride_input = input.stride()
    stride_weight = weight.stride()
    stride_out = out.stride()
    BLOCK_SIZE = triton.next_power_of_2(n_features)
    grid = lambda meta: (triton.cdiv(n_features, meta['BLOCK_SIZE']), )
    log_softmax_linear_kernel[grid](input, weight, bias, out, stride_input,
                                     stride_weight, stride_out, n_features,
                                     out_features, last_dim, BLOCK_SIZE)
    if last_dim != -1:
        out = out * (-1)
    return out
