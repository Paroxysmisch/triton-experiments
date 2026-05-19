import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, bias_shape, output_shape,
    block_size=256):
    
    pid = tl.program_id(axis=0)
    coords = pid * block_size + tl.arange(0, block_size)

    # Load input and weight
    input_data = tl.load(input_ptr + coords[:, None], mask=coords < input_shape[0])
    weight_data = tl.load(weight_ptr + coords[None, :], mask=coords < weight_shape[1])

    # Compute linear transformation
    lin_result = tl.dot(input_data, weight_data.T, allow_tf32=False)

    # Add bias if provided
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr + coords[:, None], mask=coords < bias_shape[0])
        lin_result += bias_data

    # LogSoftmax computation
    max_val = tl.max(lin_result, axis=1, keepdims=True)
    exp_vals = tl.exp(lin_result - max_val)
    sum_exp = tl.sum(exp_vals, axis=1, keepdims=True)
    softmax = exp_vals / sum_exp
    log_softmax = lin_result - max_val - tl.log(sum_exp)

    # Store the result
    tl.store(output_ptr + coords[:, None], log_softmax, mask=coords < output_shape[0])

# Triton C++ Wrapper
import torch
from torch.autograd import Function

class LogSoftmaxLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, dim=-1, dtype=None):
        if dtype is not None:
            input = input.to(dtype=dtype)

        if bias is not None:
            assert bias.shape == (weight.shape[0],), "Bias must have shape (out_features)"
        
        output_shape = list(input.shape)
        output_shape[dim] = weight.shape[0]

        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

        grid = lambda meta: (triton.cdiv(output_shape[0], meta['block_size']),)

        log_softmax_kernel[grid](input.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None, output.data_ptr(),
                                  input.shape, weight.shape, bias.shape if bias is not None else None, output.shape,
                                  block=(block_size, 1, 1))

        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        batch_size, in_features = input.shape
        out_features = weight.shape[0]

        for i in range(batch_size):
            for j in range(in_features):
                for k in range(out_features):
                    if bias is not None:
                        grad_weight[k, j] += (grad_output[i, k] * output[i, k]) * input[i, j]
                        grad_bias[k] += grad_output[i, k] * output[i, k]
                    else:
                        grad_weight[k, j] += (grad_output[i, k] * output[i, k]) * input[i, j]

        return grad_input, grad_weight, grad_bias, None, None

log_softmax_linear = LogSoftmaxLinearFunction.apply
