import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    x_w1_ptr, weight2_ptr, bias_ptr, output_ptr,
    B, N, D_out,
    stride_x_b, stride_x_n, stride_x_d,
    stride_w2_b, stride_w2_n, stride_w2_d,
    stride_bias_b, stride_bias_n, stride_bias_d,
    stride_output_b, stride_output_n, stride_output_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * N * D_out
    
    k = offsets % D_out
    j = (offsets // D_out) % N
    i = offsets // (D_out * N)
    
    x_offset = i * stride_x_b + j * stride_x_n + k * stride_x_d
    x = tl.load(x_w1_ptr + x_offset, mask=mask, other=0.0)
    
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    tanh_sig_x = tl.tanh(sig_x)
    
    w2_offset = i * stride_w2_b + j * stride_w2_n + k * stride_w2_d
    w2 = tl.load(weight2_ptr + w2_offset, mask=mask, other=0.0)
    
    temp = tanh_sig_x * w2
    
    bias_offset = i * stride_bias_b + j * stride_bias_n + k * stride_bias_d
    bias = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
    
    output_val = temp + bias
    
    output_offset = i * stride_output_b + j * stride_output_n + k * stride_output_d
    tl.store(output_ptr + output_offset, output_val, mask=mask)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Perform matrix multiplication
    x_w1 = torch.matmul(input, weight1)
    original_shape = x_w1.shape
    B_flat, N, D_out = x_w1.view(-1, *original_shape[-2:]).shape
    
    x_w1_flat = x_w1.view(B_flat, N, D_out)
    
    if out is None:
        output = torch.empty_like(x_w1)
    else:
        assert out.shape == original_shape
        output = out
    output_flat = output.view(B_flat, N, D_out)
    
    def get_strides(tensor, expand_shape):
        expanded = tensor.expand(expand_shape)
        return expanded.stride(0), expanded.stride(1), expanded.stride(2)
    
    expand_shape = (B_flat, N, D_out)
    w2_stride = get_strides(weight2, expand_shape)
    bias_stride = get_strides(bias, expand_shape)
    
    x_stride = (x_w1_flat.stride(0), x_w1_flat.stride(1), x_w1_flat.stride(2))
    output_stride = (output_flat.stride(0), output_flat.stride(1), output_flat.stride(2))
    
    num_elements = B_flat * N * D_out
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    
    combined_activation_kernel[grid](
        x_w1_flat, weight2, bias, output_flat,
        B_flat, N, D_out,
        x_stride[0], x_stride[1], x_stride[2],
        w2_stride[0], w2_stride[1], w2_stride[2],
        bias_stride[0], bias_stride[1], bias_stride[2],
        output_stride[0], output_stride[1], output_stride[2],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
