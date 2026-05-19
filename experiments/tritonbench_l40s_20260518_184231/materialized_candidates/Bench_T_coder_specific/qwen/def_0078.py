import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    n, c,
    dim,
    weight_ptr,
    ignore_index,
    reduction,
    label_smoothing,
    stride_n,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n)

    # Load input and target data
    x = tl.load(input_ptr + block_start * stride_n + pid * stride_c, mask=block_end <= n, other=float('-inf'))
    y = tl.load(target_ptr + block_start, mask=block_end <= n, other=ignore_index)

    # Apply log softmax
    max_x = tl.max(x, axis=dim)
    exp_x = tl.exp(x - max_x)
    sum_exp_x = tl.sum(exp_x, axis=dim)
    log_sum_exp_x = max_x + tl.log(sum_exp_x)
    log_softmax_x = x - log_sum_exp_x

    # Apply label smoothing
    if label_smoothing > 0.0:
        smooth_target = (1.0 - label_smoothing) * tl.one_hot(y, num_classes=c, dtype=x.dtype) + label_smoothing / c
    else:
        smooth_target = tl.one_hot(y, num_classes=c, dtype=x.dtype)

    # Compute cross-entropy loss
    ce_loss = -tl.reduce(tl.sum(log_softmax_x * smooth_target, axis=dim), axis=0, op=tl.math.sum)

    # Apply weight if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + pid * stride_c, mask=True, other=1.0)
        ce_loss *= weight

    # Reduce loss based on reduction method
    if reduction == 'mean':
        ce_loss /= n
    elif reduction == 'sum':
        pass

    # Store result
    tl.store(output_ptr + pid, ce_loss, mask=pid < n)
