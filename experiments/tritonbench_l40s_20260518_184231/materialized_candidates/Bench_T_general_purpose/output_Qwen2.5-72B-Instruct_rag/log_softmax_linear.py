import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_input_n, stride_input_d, stride_weight_m, stride_weight_n,
    stride_output_m, stride_output_n,
    N, M, K, DIM,
    USE_BIAS: tl.constexpr,
    DIM_IS_LAST: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_size_m = tl.cdiv(M, tl.num_programs(axis=0))
    block_start = pid * block_size_m
    offsets_m = block_start + tl.arange(0, block_size_m)
    offsets_n = tl.arange(0, K)
    mask = (offsets_m < M)[:, None] & (offsets_n < K)[None, :]

    # Load input and weight
    input = tl.load(input_ptr + offsets_n[None, :] * stride_input_n, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets_m[:, None] * stride_weight_m + offsets_n[None, :] * stride_weight_n, mask=mask, other=0.0)

    # Compute linear transformation
    acc = tl.zeros((block_size_m, 1), dtype=tl.float32)
    for k in range(0, K, 32):
        a = input[:, k:k+32]
        b = weight[:, k:k+32]
        acc += tl.dot(a, b.T)

    # Apply bias if provided
    if USE_BIAS:
        bias = tl.load(bias_ptr + offsets_m, mask=offsets_m < M, other=0.0)
        acc += bias[:, None]

    # Compute log_softmax
    max_val = tl.max(acc, axis=0)
    numerator = acc - max_val
    exp_num = tl.exp(numerator)
    sum_exp = tl.sum(exp_num, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val
    output = numerator - log_sum_exp

    # Store the result
    tl.store(output_ptr + offsets_m[:, None] * stride_output_m + offsets_n[None, :] * stride_output_n, output, mask=mask)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
        weight = weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)

    N, K = input.shape
    M, _ = weight.shape

    if dim != -1:
        input = input.transpose(dim, -1)
        dim = -1

    output = torch.empty((N, M), device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(M, 128),)
    log_softmax_linear_kernel[grid](
        input, weight, bias if bias is not None else torch.zeros((M,), device=input.device, dtype=input.dtype),
        output,
        input.stride(0), input.stride(1), weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        N, M, K, dim,
        USE_BIAS=bias is not None,
        DIM_IS_LAST=True
    )

    if dim != -1:
        output = output.transpose(dim, -1)

    return output
