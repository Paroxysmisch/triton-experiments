import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    logits_ptr, targets_ptr, output_ptr, loss_ptr,
    normalized_shape, weight_ptr, ignore_index, reduction, label_smoothing, eps,
    N, C, stride_logits, stride_targets, stride_output, stride_loss
):
    # Define block size for parallelization
    BLOCK_SIZE = 128

    # Calculate the indices for this program
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load logits and targets
    logits = tl.load(logits_ptr + offsets * stride_logits, mask=offsets < N * C, other=0.0)
    targets = tl.load(targets_ptr + offsets * stride_targets, mask=offsets < N, other=-1)

    # Softmax computation
    max_logits = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp_logits

    # Cross-entropy loss computation
    if reduction == 'none':
        loss = -tl.log(probs) * (targets != ignore_index)
    else:
        # Apply label smoothing if needed
        smooth_target = (1.0 - label_smoothing) * targets + label_smoothing / C
        loss = -tl.sum(smooth_target * tl.log(probs), axis=0) * (targets != ignore_index)

    # Layer normalization
    mean_probs = tl.mean(probs, axis=0)
    var_probs = tl.var(probs, axis=0)
    norm_probs = (probs - mean_probs) / tl.sqrt(var_probs + eps)

    # Store the results
    tl.store(output_ptr + offsets * stride_output, norm_probs, mask=offsets < N * C)
    tl.store(loss_ptr + offsets * stride_loss, loss, mask=offsets < N)

    # Apply class weights if provided
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + offsets, mask=offsets < C, other=1.0)
        loss *= weights

    # Reduce the loss if necessary
    if reduction == 'mean':
        loss = tl.sum(loss) / N
    elif reduction == 'sum':
        loss = tl.sum(loss)

    # Store the final loss
    tl.store(loss_ptr + offsets * stride_loss, loss, mask=offsets < N)

import torch
import triton
import triton.language as tl

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100,
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    # Ensure logits and targets are on the same device
    assert logits.device == targets.device, "Logits and targets must be on the same device"
    
    # Determine the shape parameters
    N, C = logits.shape[:2]
    stride_logits = logits.stride(0)
    stride_targets = targets.stride(0)
    
    # Prepare output tensors
    if out is None:
        out = torch.empty_like(logits)
    loss = torch.empty(N, device=logits.device, dtype=logits.dtype)
    
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(N * C, META['BLOCK_SIZE']),)
    fused_kernel[grid](
        logits, targets, out, loss,
        normalized_shape, weight, ignore_index, reduction, label_smoothing, eps,
        N, C, stride_logits, stride_targets, out.stride(0), loss.stride(0)
    )
    
    return loss, out
