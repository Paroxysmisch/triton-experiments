import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr, target_ptr, output_ptr, weight_ptr, ignore_index, reduction, label_smoothing,
    n_elements, n_classes, stride_input, stride_target, stride_output, stride_weight,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input and target
    input_offsets = offsets[:, None] * stride_input + tl.arange(0, n_classes)
    target_offsets = offsets * stride_target

    input = tl.load(input_ptr + input_offsets, mask=offsets < n_elements, other=0.0)
    target = tl.load(target_ptr + target_offsets, mask=offsets < n_elements, other=ignore_index)

    # Apply label smoothing
    if label_smoothing > 0.0:
        smooth_target = (1.0 - label_smoothing) * tl.equal(target[:, None], tl.arange(0, n_classes)) + label_smoothing / n_classes
    else:
        smooth_target = tl.equal(target[:, None], tl.arange(0, n_classes))

    # Compute log softmax
    max_val = tl.max(input, axis=1)[:, None]
    input = input - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(input), axis=1))[:, None]
    log_softmax = input - log_sum_exp

    # Compute cross entropy loss
    loss = -tl.sum(smooth_target * log_softmax, axis=1)

    # Apply weights
    if weight_ptr is not None:
        weight_offsets = target * stride_weight
        weights = tl.load(weight_ptr + weight_offsets, mask=offsets < n_elements, other=1.0)
        loss = loss * weights

    # Apply reduction
    if reduction == 0:  # 'none'
        tl.store(output_ptr + offsets, loss, mask=offsets < n_elements)
    elif reduction == 1:  # 'mean'
        loss_sum = tl.sum(loss, axis=0)
        loss_count = tl.sum(smooth_target, axis=0)
        loss_mean = loss_sum / loss_count
        tl.store(output_ptr, loss_mean)
    elif reduction == 2:  # 'sum'
        loss_sum = tl.sum(loss, axis=0)
        tl.store(output_ptr, loss_sum)

import torch
import triton

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Ensure input and target are on the same device
    device = input.device
    target = target.to(device)

    # Get dimensions
    n_elements = input.size(0)
    n_classes = input.size(dim)

    # Reshape input and target for Triton kernel
    input = input.view(n_elements, n_classes)
    target = target.view(n_elements)

    # Initialize output tensor
    if reduction == 'none':
        output = torch.empty_like(input, device=device)
    else:
        output = torch.empty(1, device=device)

    # Initialize weight tensor if provided
    if weight is not None:
        weight = weight.to(device)
    else:
        weight = None

    # Define reduction method
    reduction_map = {'none': 0, 'mean': 1, 'sum': 2}
    reduction_code = reduction_map[reduction]

    # Launch Triton kernel
    grid = (n_elements // 1024 + 1,)
    fused_cross_entropy_log_softmax_kernel[grid](
        input, target, output, weight, ignore_index, reduction_code, label_smoothing,
        n_elements, n_classes, input.stride(0), target.stride(0), output.stride(0), weight.stride(0) if weight is not None else 0,
        BLOCK_SIZE=1024
    )

    return output
