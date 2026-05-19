import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    logits_ptr,  # Pointer to logits tensor
    target_ptr,  # Pointer to target distribution tensor
    weight_ptr,  # Pointer to class weights tensor, can be null
    mask_ptr,    # Pointer to mask tensor indicating ignored samples
    output_ptr,  # Pointer to output loss tensor
    n_samples,   # Number of samples (N)
    n_classes,   # Number of classes (C)
    logits_row_stride, logits_col_stride,  # Strides for logits tensor
    target_row_stride, target_col_stride,  # Strides for target tensor
    weight_stride,  # Stride for weights tensor
    mask_stride,    # Stride for mask tensor
    BLOCK_SIZE: tl.constexpr,  # Number of samples processed per block
):
    pid = tl.program_id(0)
    sample_start = pid * BLOCK_SIZE
    offsets = sample_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples

    # Load mask for current block
    mask_offsets = offsets * mask_stride
    ignore_mask = tl.load(mask_ptr + mask_offsets, mask=mask, other=0)

    # Initialize logits and max values
    row_max = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    logits = tl.zeros((BLOCK_SIZE, n_classes), dtype=tl.float32)

    # Load logits and compute max
    for c in range(n_classes):
        ptrs = logits_ptr + offsets[:, None] * logits_row_stride + c * logits_col_stride
        cols_mask = mask[:, None] & (c < n_classes)
        val = tl.load(ptrs, mask=cols_mask, other=0)
        logits = tl.where(cols_mask, val, logits)
        row_max = tl.maximum(row_max, val)

    # Subtract max for numerical stability
    logits -= row_max[:, None]

    # Compute exp and sum
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=1)
    log_sum_exp = tl.log(sum_exp)

    # Compute log_softmax
    log_softmax = logits - log_sum_exp[:, None]

    # Compute cross entropy loss
    loss = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(n_classes):
        # Load target and weight
        target_ptrs = target_ptr + offsets[:, None] * target_row_stride + c * target_col_stride
        target = tl.load(target_ptrs, mask=mask[:, None] & (c < n_classes), other=0).to(tl.float32)
        log_p = log_softmax[:, c]

        if weight_ptr is not None:
            weight = tl.load(weight_ptr + c * weight_stride).to(tl.float32)
            term = weight * target * log_p
        else:
            term = target * log_p

        loss += term

    # Apply negative and mask
    loss = -loss
    loss = tl.where(ignore_mask, 0.0, loss)

    # Store result
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, loss, mask=mask)

def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    # Ensure input is 2D and dim is valid
    assert input.dim() == 2, "Input must be 2D"
    N, C = input.shape
    assert 0 <= dim < input.dim(), "Invalid dimension"

    # Handle dimension permutation if necessary
    if dim != 1:
        input = input.transpose(1, dim).contiguous()
        permuted = True
    else:
        permuted = False

    # Process target based on type
    if target.dtype in (torch.long, torch.int):
        assert target.shape == (N,), "Target shape mismatch for class indices"
        mask = target == ignore_index
        target_dist = torch.full((N, C), label_smoothing / C, dtype=input.dtype, device=input.device)
        valid_targets = target[~mask]
        target_dist[~mask] = target_dist[~mask].scatter_(1, valid_targets.unsqueeze(1), 1 - label_smoothing + label_smoothing / C)
    else:
        assert target.shape == (N, C), "Target shape mismatch for distribution"
        mask = torch.zeros(N, dtype=torch.bool, device=input.device)
        target_dist = target

    # Ensure tensors are contiguous
    input = input.contiguous()
    target_dist = target_dist.contiguous()
    mask = mask.contiguous()

    # Allocate output
    output = torch.empty_like(input[:, 0], dtype=torch.float32)

    # Kernel configuration
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_cross_entropy_log_softmax_kernel[grid](
        input, target_dist, weight, mask, output,
        N, C,
        input.stride(0), input.stride(1),
        target_dist.stride(0), target_dist.stride(1),
        weight.stride(0) if weight is not None else 0,
        mask.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Undo permutation if necessary
    if permuted:
        input = input.transpose(1, dim)

    # Apply reduction
    if reduction == 'none':
        return output
    sum_loss = output.sum()
    if reduction == 'sum':
        return sum_loss
    num_non_ignored = N - mask.sum()
    if num_non_ignored == 0:
        return torch.tensor(0.0, device=input.device)
    return sum_loss / num_non_ignored
