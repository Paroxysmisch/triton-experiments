import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, probs_ptr, mean_ptr, inv_std_ptr, layernorm_output_ptr,
    logits_shape, targets_shape, normalized_shape, weight_ptr, ignore_index, reduction,
    label_smoothing, eps, num_warps, num_stages
):
    # Unpack shapes
    batch_size, num_classes = logits_shape
    _, *extra_dims = logits_shape

    # Get global index
    pid = tl.program_id(axis=0)
    block_start = pid * tl.block_size.x
    offsets = block_start + tl.arange(0, tl.block_size.x)

    # Load logits and targets
    logits = tl.load(logits_ptr + offsets, mask=(offsets < batch_size), eviction_policy=tl.EVICT_FALSE)
    targets = tl.load(targets_ptr + offsets, mask=(offsets < batch_size), eviction_policy=tl.EVICT_FALSE)

    # Apply label smoothing
    if label_smoothing > 0.0:
        smooth_targets = (1.0 - label_smoothing) * targets + label_smoothing / num_classes
    else:
        smooth_targets = targets

    # Compute cross-entropy loss
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1, keepdim=True)
    probs = exp_logits / sum_exp_logits
    log_probs = tl.log(probs)
    ce_loss = -tl.sum(smooth_targets * log_probs, axis=1)

    # Softmax activation
    # Already computed above

    # Layer normalization
    mean = tl.mean(probs, axis=1, keepdim=True)
    diff = probs - mean
    inv_std = tl.rsqrt(tl.mean(diff * diff, axis=1, keepdim=True) + eps)
    normalized_probs = diff * inv_std

    # Store results
    tl.store(probs_ptr + offsets, probs, mask=(offsets < batch_size))
    tl.store(mean_ptr + offsets, mean, mask=(offsets < batch_size))
    tl.store(inv_std_ptr + offsets, inv_std, mask=(offsets < batch_size))
    tl.store(layernorm_output_ptr + offsets, normalized_probs, mask=(offsets < batch_size))

    # Reduction
    if reduction == 'mean':
        total_ce_loss = tl.sum(ce_loss, axis=0)
        total_ce_loss /= batch_size
        tl.store(out_ptr, total_ce_loss, mask=(pid == 0))
