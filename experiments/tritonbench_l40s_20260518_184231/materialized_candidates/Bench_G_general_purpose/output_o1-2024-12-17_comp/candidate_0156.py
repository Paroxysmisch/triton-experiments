import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['D']
)
@triton.heuristics({
    'IS_FP16': lambda *args, **kwargs: args[0].dtype == tl.float16,
    'DEPTH': lambda *args, **kwargs: tl.next_power_of_2(args[3])
})
@triton.jit
def _softmax(
    input_ptr, output_ptr,
    B, S, D,
    stride_b, stride_s, stride_d,
    LOG: tl.constexpr, CAUSAL: tl.constexpr, MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr, DEPTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_SIZE)
    offs = pid_b * stride_b + pid_s * stride_s + offs_d
    # Load
    mask = offs_d < D
    x = tl.where(mask, tl.load(input_ptr + offs, mask=mask), float('-inf'))
    if CAUSAL:
        causal_mask = offs_d > pid_s
        x = tl.where(causal_mask, float('-inf'), x)
    if MASK_TYPE:
        # example masked out values, the real mask logic can be extended
        mask_cond = offs_d % 2 == 0
        x = tl.where(mask_cond, float('-inf'), x)

    # Compute max
    x_max = tl.max(x, 0)
    # Sub
    x = x - x_max
    # Exp
    if LOG:
        exp_x = tl.exp(x)
        exp_sum = tl.sum(exp_x, 0)
        x = x - tl.log(exp_sum)
    else:
        exp_x = tl.exp(x)
        exp_sum = tl.sum(exp_x, 0)
        x = exp_x / exp_sum

    # Store
    tl.store(output_ptr + offs, x, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['D']
)
@triton.heuristics({
    'IS_FP16': lambda *args, **kwargs: args[0].dtype == tl.float16,
    'DEPTH': lambda *args, **kwargs: tl.next_power_of_2(args[3])
})
@triton.jit
def _softmax_backward(
    grad_ptr, output_ptr, dinput_ptr,
    B, S, D,
    stride_b, stride_s, stride_d,
    LOG: tl.constexpr, CAUSAL: tl.constexpr, MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr, DEPTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_SIZE)
    offs = pid_b * stride_b + pid_s * stride_s + offs_d
    # Load
    mask = offs_d < D
    grad_val = tl.where(mask, tl.load(grad_ptr + offs, mask=mask), 0.0)
    out_val = tl.where(mask, tl.load(output_ptr + offs, mask=mask), 0.0)

    if CAUSAL:
        causal_mask = offs_d > pid_s
        grad_val = tl.where(causal_mask, 0.0, grad_val)
        out_val = tl.where(causal_mask, 0.0, out_val)
    if MASK_TYPE:
        mask_cond = offs_d % 2 == 0
        grad_val = tl.where(mask_cond, 0.0, grad_val)
        out_val = tl.where(mask_cond, 0.0, out_val)

    if LOG:
        # For log softmax: grad_input = grad_output - exp(logits)*sum(grad_output)
        sum_grad = tl.sum(grad_val, 0)
        dinput = grad_val - tl.exp(out_val) * sum_grad
    else:
        # For softmax: grad_input = output * (grad_output - sum(grad_output*output))
        sum_grad = tl.sum(grad_val * out_val, 0)
        dinput = out_val * (grad_val - sum_grad)

    tl.store(dinput_ptr + offs, dinput, mask=mask)

def softmax(input, log=False, causal=False, mask_type=False):
    import math
    B, S, D = input.shape
    output = input.new_empty(input.shape)
    grid = (B, S)
    BLOCK_SIZE = 1024 if D > 1024 else 256
    _softmax[grid](
        input, output,
        B, S, D,
        input.stride(0), input.stride(1), input.stride(2),
        log, causal, mask_type,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def softmax_backward(grad, output, log=False, causal=False, mask_type=False):
    B, S, D = grad.shape
    dinput = grad.new_empty(grad.shape)
    grid = (B, S)
    BLOCK_SIZE = 1024 if D > 1024 else 256
    _softmax_backward[grid](
        grad, output, dinput,
        B, S, D,
        grad.stride(0), grad.stride(1), grad.stride(2),
        log, causal, mask_type,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dinput
