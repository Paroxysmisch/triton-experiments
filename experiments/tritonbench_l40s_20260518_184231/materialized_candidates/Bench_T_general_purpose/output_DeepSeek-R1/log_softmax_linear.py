import torch
import triton
import triton.language as tl

@triton.jit
def _log_softmax_linear_fused_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    input_row_stride,
    weight_row_stride,
    bias_stride,
    output_row_stride,
    dtype: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid_row = tl.program_id(0)
    num_pid_rows = tl.num_programs(0)

    row_idx = pid_row
    if row_idx >= num_pid_rows:
        return

    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    max_val = -float('inf')
    sum_exp = 0.0
    linear_accumulator = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)

    for out_feat_block in range(0, out_features, BLOCK_SIZE_OUT):
        out_feat_index = out_feat_block + tl.arange(0, BLOCK_SIZE_OUT)
        mask_out = out_feat_index < out_features

        linear_val = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
        for in_feat_block in range(0, in_features, BLOCK_SIZE_IN):
            in_feat_index = in_feat_block + tl.arange(0, BLOCK_SIZE_IN)
            mask_in = in_feat_index < in_features

            input_val = tl.load(input_row_ptr + in_feat_index, mask=mask_in, other=0.0)
            weight_offset = out_feat_index[:, None] * weight_row_stride + in_feat_index[None, :]
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask_out[:, None] & mask_in[None, :], other=0.0)

            linear_val += tl.sum(input_val[None, :] * weight_val, axis=1)

        if HAS_BIAS:
            bias_val = tl.load(bias_ptr + out_feat_index, mask=mask_out, other=0.0)
            linear_val += bias_val

        current_max = tl.max(linear_val, axis=0)
        max_val = tl.maximum(max_val, current_max)

        shifted = linear_val - max_val
        exp_shifted = tl.exp(shifted)
        sum_exp += tl.sum(exp_shifted, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val

    for out_feat_block in range(0, out_features, BLOCK_SIZE_OUT):
        out_feat_index = out_feat_block + tl.arange(0, BLOCK_SIZE_OUT)
        mask_out = out_feat_index < out_features

        linear_val = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
        for in_feat_block in range(0, in_features, BLOCK_SIZE_IN):
            in_feat_index = in_feat_block + tl.arange(0, BLOCK_SIZE_IN)
            mask_in = in_feat_index < in_features

            input_val = tl.load(input_row_ptr + in_feat_index, mask=mask_in, other=0.0)
            weight_offset = out_feat_index[:, None] * weight_row_stride + in_feat_index[None, :]
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask_out[:, None] & mask_in[None, :], other=0.0)

            linear_val += tl.sum(input_val[None, :] * weight_val, axis=1)

        if HAS_BIAS:
            bias_val = tl.load(bias_ptr + out_feat_index, mask=mask_out, other=0.0)
            linear_val += bias_val

        log_softmax_val = linear_val - log_sum_exp
        output_offset = out_feat_index
        tl.store(output_row_ptr + output_offset, log_softmax_val.to(dtype), mask=mask_out)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    assert dim == -1 or dim == input.dim() - 1, "log_softmax must be applied along the last dimension"
    if dtype is not None:
        input = input.to(dtype)
    else:
        dtype = input.dtype

    in_features = input.size(-1)
    out_features = weight.size(0)
    assert weight.size(1) == in_features, "Expected weight size (out_features, in_features)"
    output_shape = input.shape[:-1] + (out_features,)
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    input_row_stride = input.stride(-2)
    weight_row_stride = weight.stride(0)
    bias_stride = bias.stride(0) if bias is not None else 0
    output_row_stride = output.stride(-2)

    num_rows = input.numel() // in_features
    grid = lambda meta: (num_rows,)
    BLOCK_SIZE_IN = 32
    BLOCK_SIZE_OUT = 32

    _log_softmax_linear_fused_kernel[grid](
        input,
        weight,
        bias if bias is not None else torch.empty(0, device=input.device),
        output,
        in_features,
        out_features,
        input_row_stride,
        weight_row_stride,
        bias_stride,
        output_row_stride,
        tl.from_dtype(dtype),
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT,
    )
    return output
