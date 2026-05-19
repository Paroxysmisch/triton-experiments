import triton
import triton.language as tl

# Triton kernel code
@triton.jit
def cross_entropy_fwd_kernel(
    logits,
    labels,
    weights,
    lse,
    smoothed_loss,
    num_classes,
    batch_size,
    smoothing,
    lse_square_scale,
    ignored_index,
    block_size,
    grid_size,
):
    idx = tl.program_id(0)
    class_idx = idx % num_classes
    batch_idx = idx // num_classes
    
    if batch_idx >= batch_size:
        return
    
    label = labels[batch_idx]
    weight = weights[batch_idx]
    
    if label == ignored_index:
        return
    
    logit = logits[batch_idx * num_classes + class_idx]
    max_logit = logits[batch_idx * num_classes + label]
    
    exp_sum = 0.0
    for i in range(num_classes):
        exp_sum += tl.exp(logits[batch_idx * num_classes + i] - max_logit)
    
    lse_value = tl.log(exp_sum) + max_logit
    lse[batch_idx] = lse_value
    
    smooth_loss = 0.0
    if class_idx == label:
        smooth_loss = -tl.log(exp(logit - max_logit) / exp_sum) * weight
    else:
        smooth_loss = -tl.log(exp(logit - max_logit) / exp_sum) * (1.0 - smoothing) * weight
        smooth_loss += -tl.log(exp(logit - max_logit) / exp_sum) * smoothing / (num_classes - 1)
    
    smoothed_loss[batch_idx] = smooth_loss * lse_square_scale

@triton.jit
def cross_entropy_bwd_kernel(
    logits,
    labels,
    weights,
    lse,
    smoothed_loss,
    grad_logits,
    num_classes,
    batch_size,
    smoothing,
    ignored_index,
    block_size,
    grid_size,
):
    idx = tl.program_id(0)
    class_idx = idx % num_classes
    batch_idx = idx // num_classes
    
    if batch_idx >= batch_size:
        return
    
    label = labels[batch_idx]
    weight = weights[batch_idx]
    
    if label == ignored_index:
        return
    
    logit = logits[batch_idx * num_classes + class_idx]
    max_logit = logits[batch_idx * num_classes + label]
    
    exp_sum = 0.0
    for i in range(num_classes):
        exp_sum += tl.exp(logits[batch_idx * num_classes + i] - max_logit)
    
    grad_logit = 0.0
    if class_idx == label:
        grad_logit = (tl.exp(logit - max_logit) / exp_sum - 1.0) * weight
    else:
        grad_logit = (tl.exp(logit - max_logit) / exp_sum) * (1.0 - smoothing) * weight
        grad_logit -= (tl.exp(logit - max_logit) / exp_sum) * smoothing / (num_classes - 1)
    
    grad_logits[batch_idx * num_classes + class_idx] = grad_logit * smoothed_loss[batch_idx]

# Triton wrapper function
def cross_entropy_loss(logits, labels, smoothing=0.0, lse_square_scale=1.0, ignored_index=-100):
    num_classes = logits.shape[1]
    batch_size = logits.shape[0]
    
    lse = tl.zeros((batch_size,), dtype=tl.float32)
    smoothed_loss = tl.zeros((batch_size,), dtype=tl.float32)
    grad_logits = tl.zeros_like(logits)
    
    # Launch forward kernel
    grid_size = (num_classes * batch_size + triton.cdiv(block_size, 1) - 1) // triton.cdiv(block_size, 1)
    cross_entropy_fwd_kernel[grid_size, block_size](
        logits,
        labels,
        tl.zeros((batch_size,), dtype=tl.float32),
        lse,
        smoothed_loss,
        num_classes,
        batch_size,
        smoothing,
        lse_square_scale,
        ignored_index,
        block_size,
        grid_size,
    )
    
    # Launch backward kernel
    cross_entropy_bwd_kernel[grid_size, block_size](
        logits,
        labels,
        tl.zeros((batch_size,), dtype=tl.float32),
        lse,
        smoothed_loss,
        grad_logits,
        num_classes,
        batch_size,
        smoothing,
        ignored_index,
        block_size,
        grid_size,
    )
    
    return smoothed_loss.sum(), grad_logits

# Example usage
logits = tl.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]], dtype=tl.float32)
labels = tl.tensor([0, 1], dtype=tl.int32)
loss, grad = cross_entropy_loss(logits, labels, smoothing=0.1)
print("Loss:", loss)
print("Gradient:", grad)
