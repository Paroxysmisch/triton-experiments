import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def batchnorm_relu_kernel(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr,
    eps,
    n, c, h, w,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = n * c * h * w
    for idx in range(pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE):
        if idx >= num_elements:
            return
        
        # Calculate indices
        idx_n = idx // (c * h * w)
        remainder = idx % (c * h * w)
        idx_c = remainder // (h * w)
        remainder = remainder % (h * w)
        idx_h = remainder // w
        idx_w = remainder % w

        # Compute pointers
        input_offset = idx_n * input_stride_n + idx_c * input_stride_c + idx_h * input_stride_h + idx_w * input_stride_w
        output_offset = idx_n * output_stride_n + idx_c * output_stride_c + idx_h * output_stride_h + idx_w * output_stride_w

        x = tl.load(input_ptr + input_offset)
        mean = tl.load(running_mean_ptr + idx_c)
        var = tl.load(running_var_ptr + idx_c)
        gamma = tl.load(gamma_ptr + idx_c) if gamma_ptr is not None else 1.0
        beta = tl.load(beta_ptr + idx_c) if beta_ptr is not None else 0.0

        inv_std = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * inv_std * gamma + beta
        y_relu = tl.maximum(y, 0.0)
        tl.store(output_ptr + output_offset, y_relu)

def apply_batchnorm_relu_triton(input, running_mean, running_var, gamma, beta, eps, inplace):
    if not input.is_contiguous():
        input = input.contiguous()
    n, c, h, w = input.shape
    output = torch.empty_like(input) if not inplace else input

    gamma = torch.ones(c, device=input.device, dtype=input.dtype) if gamma is None else gamma
    beta = torch.zeros(c, device=input.device, dtype=input.dtype) if beta is None else beta

    num_elements = n * c * h * w
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    batchnorm_relu_kernel[grid](
        input, output,
        running_mean, running_var,
        gamma, beta,
        eps,
        n, c, h, w,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=1024,
    )
    return output

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Step 1: Perform convolution
    conv_output = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # Step 2: Apply BatchNorm and ReLU
    if training or running_mean is None or running_var is None:
        # Use PyTorch's native functions for training or missing running stats
        bn_output = F.batch_norm(
            conv_output,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=momentum,
            eps=eps
        )
        output = F.relu(bn_output, inplace=inplace)
    else:
        # Use Triton kernel for inference
        output = apply_batchnorm_relu_triton(
            conv_output,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            eps,
            inplace
        )
    
    return output
