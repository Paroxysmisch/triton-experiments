import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    weight_ptr,
    num_classes,
    batch_size,
    dim,
    ignore_index,
    reduction,
    label_smoothing,
    stride_input,
    stride_target,
    stride_output,
    block_size: tl.constexpr,
):
    # Determine the index of the current thread within the block
    pid = tl.program_id(axis=0)
    coords = tl.program_id(axis=1)

    # Get the start index for the current thread
    base_idx = pid * block_size + coords

    # Load the relevant parts of the input and target tensors
    input_block = tl.load(input_ptr + base_idx * stride_input, num_elements=block_size)
    target_block = tl.load(target_ptr + base_idx * stride_target, num_elements=block_size)

    # Apply label smoothing
    if label_smoothing > 0.0:
        smooth_label = (1.0 - label_smoothing) * target_block + label_smoothing / num_classes
    else:
        smooth_label = target_block

    # Compute log softmax
    max_val = tl.max(input_block, axis=dim)
    exp_values = tl.exp(input_block - max_val[:, None])
    sum_exp = tl.sum(exp_values, axis=dim)
    log_probs = input_block - max_val[:, None] - tl.log(sum_exp)

    # Compute cross-entropy loss
    ce_loss = -tl.sum(smooth_label * log_probs, axis=dim)

    # Handle ignored targets
    ignore_mask = target_block != ignore_index
    ce_loss = ce_loss * ignore_mask

    # Apply weight if provided
    if weight_ptr is not None:
        weight_block = tl.load(weight_ptr + base_idx * stride_target, num_elements=block_size)
        ce_loss = ce_loss * weight_block

    # Reduce the loss according to the specified reduction type
    if reduction == 'mean':
        ce_loss = tl.sum(ce_loss) / batch_size
    elif reduction == 'sum':
        pass  # No action needed for sum reduction

    # Store the result in the output tensor
    tl.store(output_ptr + base_idx * stride_output, ce_loss)
