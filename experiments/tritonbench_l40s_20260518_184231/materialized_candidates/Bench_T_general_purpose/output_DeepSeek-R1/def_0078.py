import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy_log_softmax_kernel(
    input_ptr, target_probs_ptr, weight_ptr, mask_ptr, output_ptr,
    N, C,
    stride_input_row, stride_input_col,
    stride_target_row, stride_target_col,
    stride_weight,
    stride_mask,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    input_row = input_ptr + pid * stride_input_row
    target_row = target_probs_ptr + pid * stride_target_row

    # Compute max for numerical stability
    row_max = tl.load(input_row, mask=tl.arange(0, C) < C, other=-float('inf'))
    for i in range(BLOCK_SIZE, C, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < C
        val = tl.load(input_row + cols, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, val)
    row_max = tl.max(row_max, axis=0)

    # Compute sum(exp(x - row_max))
    sum_exp = 0.0
    for i in range(0, C, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < C
        x = tl.load(input_row + cols, mask=mask, other=0.0)
        x_shifted = x - row_max
        exp_x = tl.exp(x_shifted)
        sum_exp += tl.sum(exp_x, axis=0)
    log_sum = tl.log(sum_exp) + row_max

    # Compute log_softmax and accumulate loss
    loss = 0.0
    for i in range(0, C, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < C
        x = tl.load(input_row + cols, mask=mask, other=0.0)
        log_softmax = x - log_sum

        target = tl.load(target_row + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        loss += tl.sum(target * weight * log_softmax, axis=0)
    loss = -loss

    # Apply mask
    mask_val = tl.load(mask_ptr + pid * stride_mask)
    loss *= mask_val

    tl.store(output_ptr + pid, loss)

def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    assert input.dim() == 2, "Input must be 2D"
    if dim != 1:
        input = input.transpose(1, dim).contiguous()
    N, C = input.shape[0], input.shape[1]
    device = input.device

    if target.dim() == 1:
        assert target.shape[0] == N, "Target shape mismatch"
        target_indices = target.clone()
        mask = (target_indices != ignore_index).to(dtype=torch.float32, device=device)
        target_probs = torch.zeros(N, C, device=device)
        target_probs.scatter_(1, target_indices.unsqueeze(1).clamp_(0, C-1), 1.0)
        if label_smoothing > 0:
            target_probs = target_probs * (1 - label_smoothing) + (label_smoothing / C)
        target_probs *= mask.view(-1, 1)
    else:
        assert target.shape == (N, C), "Target shape mismatch"
        target_probs = target.clone().to(device=device)
        if label_smoothing > 0:
            target_probs = target_probs * (1 - label_smoothing) + (label_smoothing / C)
        mask = torch.ones(N, device=device, dtype=torch.float32)

    if weight is None:
        weight = torch.ones(C, device=device, dtype=torch.float32)
    else:
        weight = weight.to(device=device, dtype=torch.float32).contiguous()
        assert weight.shape[0] == C, "Weight shape mismatch"

    output = torch.empty(N, device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 256

    _cross_entropy_log_softmax_kernel[grid](
        input, target_probs, weight, mask, output,
        N, C,
        input.stride(0), input.stride(1),
        target_probs.stride(0), target_probs.stride(1),
        weight.stride(0),
        mask.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if reduction == 'none':
        return output
    sum_loss = output.sum()
    if reduction == 'sum':
        return sum_loss
    num_non_ignored = mask.sum().clamp_min(1e-6)
    return sum_loss / num_non_ignored
