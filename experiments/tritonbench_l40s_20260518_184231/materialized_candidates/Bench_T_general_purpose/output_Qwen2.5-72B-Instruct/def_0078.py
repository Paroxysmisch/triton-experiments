import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr, target_ptr, output_ptr, weight_ptr, ignore_index, label_smoothing,
    n_elements, n_classes, stride_input, stride_target, stride_output, stride_weight,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input and target
    input_offsets = offsets * stride_input
    target_offsets = offsets * stride_target
    input_mask = offsets < n_elements
    target_mask = offsets < n_elements

    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    target_vals = tl.load(target_ptr + target_offsets, mask=target_mask, other=ignore_index)

    # Compute log softmax
    max_val = tl.max(input_vals, axis=0)
    input_vals = input_vals - max_val
    exp_vals = tl.exp(input_vals)
    sum_exp = tl.sum(exp_vals, axis=0)
    log_sum_exp = tl.log(sum_exp)
    log_softmax_vals = input_vals - log_sum_exp

    # Compute cross entropy loss
    loss = tl.zeros_like(log_softmax_vals)
    for i in range(n_classes):
        class_mask = target_vals == i
        if label_smoothing > 0.0:
            smooth_target = (1.0 - label_smoothing) * class_mask + label_smoothing / n_classes
        else:
            smooth_target = class_mask
        loss += -smooth_target * log_softmax_vals

    # Apply weight if provided
    if weight_ptr is not None:
        weight_offsets = offsets * stride_weight
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=input_mask, other=1.0)
        loss *= weight_vals

    # Handle ignore_index
    ignore_mask = target_vals == ignore_index
    loss = tl.where(ignore_mask, 0.0, loss)

    # Store the loss
    output_offsets = offsets * stride_output
    tl.store(output_ptr + output_offsets, loss, mask=input_mask)

import torch
import triton
import triton.language as tl

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Ensure input and target are on the same device
    assert input.device == target.device, "Input and target must be on the same device"
    device = input.device

    # Reshape input and target for the kernel
    input = input.contiguous()
    target = target.contiguous()
    n_elements = input.size(0)
    n_classes = input.size(dim)

    # Reshape input to (N, C) and target to (N,)
    input = input.view(n_elements, n_classes)
    target = target.view(n_elements)

    # Allocate output tensor
    output = torch.empty_like(input, device=device)

    # Allocate weight tensor if provided
    if weight is not None:
        weight = weight.contiguous()
        assert weight.size(0) == n_classes, "Weight tensor must have the same number of elements as the number of classes"
    else:
        weight = torch.ones(n_classes, device=device)

    # Launch the Triton kernel
    grid = (n_elements, )
    fused_cross_entropy_log_softmax_kernel[grid](
        input, target, output, weight, ignore_index, label_smoothing,
        n_elements, n_classes, input.stride(0), target.stride(0), output.stride(0), weight.stride(0),
        BLOCK_SIZE=1024
    )

    # Apply reduction
    if reduction == 'mean':
        output = output.mean()
    elif reduction == 'sum':
        output = output.sum()
    elif reduction == 'none':
        output = output.view(input.size())
    else:
        raise ValueError("Invalid reduction method. Choose from 'mean', 'sum', or 'none'.")

    return output

# Sample data
input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32, device='cuda')
target = torch.tensor([1, 2], dtype=torch.int64, device='cuda')
weight = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device='cuda')

# Call the function
output = fused_cross_entropy_log_softmax(input, target, dim=1, weight=weight, ignore_index=-100, reduction='mean', label_smoothing=0.1)

# Print the output
print(output)
