import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    # Pointers to matrices
    logits_ptr, targets_ptr, output_probs_ptr, output_loss_ptr,
    weight_ptr, normalized_shape_size,
    # Tensor dimensions
    N, C, S,
    # Strides
    stride_logits_n, stride_logits_c,
    stride_targets_n, stride_targets_c,
    # Parameters
    is_prob_target: tl.constexpr,
    ignore_index: tl.constexpr,
    reduction: tl.constexpr,
    label_smoothing: tl.constexpr,
    eps: tl.constexpr,
    elementwise_affine: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Offsets for this block
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_s = pid_s * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_n = off_n < N
    mask_s = off_s < C

    # Load logits for current block (N, C)
    logits = tl.load(logits_ptr + off_n[:, None] * stride_logits_n + off_s[None, :] * stride_logits_c,
                     mask=mask_n[:, None] & mask_s[None, :], other=0.0)
    
    # Compute softmax
    max_logits = tl.max(logits, axis=1)
    logits -= max_logits[:, None]
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=1)[:, None]
    probs = numerator / denominator

    # Compute cross-entropy loss
    if is_prob_target:
        targets = tl.load(targets_ptr + off_n[:, None] * stride_targets_n + off_s[None, :] * stride_targets_c,
                          mask=mask_n[:, None] & mask_s[None, :], other=0.0)
        log_probs = tl.log(probs + 1e-8)
        loss_per_element = -targets * log_probs
    else:
        targets = tl.load(targets_ptr + off_n * stride_targets_n, mask=mask_n, other=ignore_index)
        targets = targets.to(tl.int32)
        valid_mask = (targets != ignore_index)
        targets = tl.where(valid_mask, targets, 0)
        if label_smoothing > 0.0:
            smooth_pos = 1.0 - label_smoothing
            smooth_neg = label_smoothing / (C - 1)
            target_mask = (off_s[None, :] == targets[:, None])
            targets_smoothed = tl.where(target_mask, smooth_pos, smooth_neg)
            log_probs = tl.log(probs + 1e-8)
            loss_per_element = -targets_smoothed * log_probs
        else:
            target_mask = (off_s[None, :] == targets[:, None])
            log_probs = tl.log(probs + 1e-8)
            loss_per_element = -log_probs * target_mask
        loss_per_element = tl.where(valid_mask[:, None], loss_per_element, 0.0)
    
    loss_per_sample = tl.sum(loss_per_element, axis=1)
    
    # Apply reduction
    if reduction == 'mean':
        count = tl.sum(valid_mask) if not is_prob_target else N * S
        loss_sum = tl.sum(loss_per_sample)
        tl.atomic_add(output_loss_ptr, loss_sum)
        if pid_s == 0 and pid_n == 0:
            tl.atomic_add(output_loss_ptr + 1, count)
    elif reduction == 'sum':
        loss_sum = tl.sum(loss_per_sample)
        tl.atomic_add(output_loss_ptr, loss_sum)
    else:  # 'none'
        tl.store(output_loss_ptr + off_n, loss_per_sample, mask=mask_n)

    # Layer normalization
    # Flatten normalized_shape dimensions (S)
    probs_flat = tl.reshape(probs, (BLOCK_SIZE_N, S))
    mean = tl.sum(probs_flat, axis=1) / S
    var = tl.sum((probs_flat - mean[:, None]) ** 2, axis=1) / S + eps
    inv_std = tl.rsqrt(var)
    
    # Normalize
    normalized = (probs_flat - mean[:, None]) * inv_std[:, None]
    if elementwise_affine:
        weight = tl.load(weight_ptr + off_s, mask=mask_s, other=1.0)
        normalized = normalized * weight[None, :]
    
    # Reshape back and store
    normalized = tl.reshape(normalized, (BLOCK_SIZE_N, BLOCK_SIZE_C))
    tl.store(output_probs_ptr + off_n[:, None] * stride_logits_n + off_s[None, :] * stride_logits_c,
             normalized, mask=mask_n[:, None] & mask_s[None, :])

def fused_cross_entropy_softmax_layernorm(
    logits: torch.Tensor, targets: torch.Tensor, normalized_shape, weight=None,
    ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, out=None
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reshape logits and targets to 2D (N, C)
    original_shape = logits.shape
    C = original_shape[-1]
    logits_2d = logits.reshape(-1, C)
    N, C = logits_2d.shape
    
    # Determine if targets are probabilities
    is_prob_target = targets.shape == logits.shape
    if is_prob_target:
        targets_2d = targets.reshape(-1, C)
    else:
        targets_2d = targets.reshape(-1)
    
    # Determine normalized_shape size
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    S = 1
    for dim in normalized_shape:
        S *= dim
    
    # Allocate outputs
    output_probs = torch.empty_like(logits_2d) if out is None else out.reshape(-1, C)
    if reduction == 'none':
        output_loss = torch.empty(N, device=logits.device, dtype=logits.dtype)
    else:
        output_loss = torch.zeros(2 if reduction == 'mean' else 1, device=logits.device, dtype=logits.dtype)
    
    # Kernel configuration
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_C = 128
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(C, BLOCK_SIZE_C))
    
    fused_cross_entropy_softmax_layernorm_kernel[grid](
        logits_2d, targets_2d, output_probs, output_loss,
        weight, S,
        N, C, S,
        logits_2d.stride(0), logits_2d.stride(1),
        targets_2d.stride(0), targets_2d.stride(1) if is_prob_target else 0,
        is_prob_target,
        ignore_index,
        reduction,
        label_smoothing,
        eps,
        weight is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    # Finalize reduction
    if reduction == 'mean':
        total_loss, count = output_loss[0], output_loss[1]
        output_loss = total_loss / count if count > 0 else total_loss.new_tensor(0.0)
    elif reduction == 'sum':
        output_loss = output_loss[0]
    
    # Reshape outputs
    output_probs = output_probs.reshape(original_shape)
    return output_loss, output_probs
