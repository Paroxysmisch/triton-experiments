import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    logits_ptr, target_ptr, weight_ptr, output_ptr,
    num_classes, num_samples, dim, ignore_index,
    label_smoothing, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Load logits and targets
    logits = tl.load(logits_ptr + offsets * num_classes, mask=offsets < num_samples)
    targets = tl.load(target_ptr + offsets, mask=offsets < num_samples)
    
    # Apply log softmax
    max_logits = tl.max(logits, axis=1, keepdim=True)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1, keepdim=True)
    log_softmax = logits - tl.log(sum_exp_logits)
    
    # Calculate cross entropy loss
    target_log_probs = tl.gather(log_softmax, targets, axis=1)
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + targets, mask=targets != ignore_index)
        target_log_probs *= weights

    if label_smoothing > 0.0:
        nll_loss = -target_log_probs * (1.0 - label_smoothing) - label_smoothing * tl.mean(log_softmax, axis=1)
    else:
        nll_loss = -target_log_probs

    # Handle ignore index
    nll_loss = tl.where(targets == ignore_index, 0.0, nll_loss)
    
    # Write to output
    tl.store(output_ptr + offsets, nll_loss, mask=offsets < num_samples)


def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Check input dimensions
    assert input.dim() == 2, "Input tensor must be 2D"
    assert target.dim() == 1, "Target tensor must be 1D"
    assert input.size(0) == target.size(0), "Input and target must have the same batch size"

    num_samples, num_classes = input.size()
    BLOCK_SIZE = 1024  # Define a block size

    # Allocate output tensor
    output = torch.empty(num_samples, device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(num_samples, meta['BLOCK_SIZE']),)
    fused_cross_entropy_log_softmax_kernel[grid](
        input, target, weight, output,
        num_classes, num_samples, dim, ignore_index,
        label_smoothing, BLOCK_SIZE=BLOCK_SIZE
    )

    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    elif reduction == 'none':
        return output
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
