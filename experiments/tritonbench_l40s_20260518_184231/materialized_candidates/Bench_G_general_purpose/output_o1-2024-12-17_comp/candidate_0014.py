import math
import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,       # [num_rows, num_cols]
    labels_ptr,       # [num_rows]
    output_ptr,       # [num_rows]
    lse_ptr,          # [num_rows]
    smoothing_ptr,    # []
    lse_square_scale, # float
    ignored_index,    # int
    num_rows,         # int
    num_cols,         # int
    row_stride,       # int
    col_stride,       # int,
    apply_smoothing,  # bool
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows

    # Load smoothing factor (scalar)
    smoothing = tl.load(smoothing_ptr)

    # For each valid row, compute partial log-sum-exp and cross-entropy
    # We'll do it in chunks of BLOCK_SIZE for columns
    # This simple version processes row by row in a single kernel instance
    row_idx = offsets
    col_range = tl.arange(0, BLOCK_SIZE)
    row_ptrs = logits_ptr + row_idx[:, None] * row_stride + col_range[None, :] * col_stride

    # Initialize partial sums
    max_logit = tl.full([BLOCK_SIZE], -float('inf'), tl.float32)
    partial_sum = tl.zeros([BLOCK_SIZE], tl.float32)

    # Loop over columns in steps of BLOCK_SIZE
    for col_block_start in range(0, num_cols, BLOCK_SIZE):
        cols = col_block_start + col_range
        col_mask = cols < num_cols
        # Load logits
        logits = tl.where(mask[:, None] & col_mask[None, :], tl.load(row_ptrs + col_range[None, :]), -float('inf'))
        # Update max-logit for stable LSE
        curr_max = tl.max(logits, 1)
        max_logit = tl.maximum(curr_max, max_logit)
        row_ptrs += BLOCK_SIZE * col_stride

    # Reset pointer
    row_ptrs = logits_ptr + row_idx[:, None] * row_stride + col_range[None, :] * col_stride

    # Second pass: compute exp(logits - max_logit)
    for col_block_start in range(0, num_cols, BLOCK_SIZE):
        cols = col_block_start + col_range
        col_mask = cols < num_cols
        logits = tl.where(mask[:, None] & col_mask[None, :], tl.load(row_ptrs + col_range[None, :]), -float('inf'))
        # scaled logits
        logits = logits - max_logit[:, None]
        exp_vals = tl.exp(logits)
        partial_sum += tl.sum(exp_vals, 1)
        row_ptrs += BLOCK_SIZE * col_stride

    # Compute LSE
    local_lse = max_logit + tl.log(partial_sum + 1e-9)
    # Store LSE if wanted for backward
    if tl.any(mask):
        tl.store(lse_ptr + offsets, local_lse, mask=mask)

    # Compute cross entropy
    # We'll just do CE row by row
    # label for each row
    label = tl.load(labels_ptr + offsets, mask=mask, other=-1)
    # ignore index
    valid_label = (label != ignored_index) & mask
    # base cross entropy
    ce = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # We'll do a loop again to find logit of the correct index
    correct_logit = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    col_mask_label = (label >= 0) & (label < num_cols) & valid_label
    correct_logit = tl.where(col_mask_label, correct_logit, 0.0)

    # Load the correct logit in a direct addressing manner
    # if label >= 0 i.e. valid
    # addressing: row_ptr + label * col_stride
    # We can do partial load
    cond_idx_ptr = logits_ptr + (row_idx * row_stride) + (label * col_stride)
    correct_logit = tl.where(
        col_mask_label,
        tl.load(cond_idx_ptr, other=0.0),
        0.0
    )
    # cross entropy part ( - correct_logit + LSE )
    raw_ce = -1.0 * correct_logit + local_lse

    if apply_smoothing:
        # smoothing distribution
        smoothed_ce = local_lse  # approximate cost if label is spread
        smoothed_ce_term = smoothing * (smoothed_ce - math.log(num_cols))
        # combine
        raw_ce = (1.0 - smoothing) * raw_ce + smoothed_ce_term

    # Square scale for LSE part
    raw_ce += 0.5 * lse_square_scale * local_lse * local_lse

    tl.store(output_ptr + offsets, tl.where(valid_label, raw_ce, 0.0), mask=mask)


@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,           # [num_rows, num_cols]
    labels_ptr,           # [num_rows]
    dloss_ptr,            # [num_rows]
    lse_ptr,              # [num_rows]
    grads_ptr,            # [num_rows, num_cols]
    smoothing_ptr,        # []
    lse_square_scale,     # float
    ignored_index,        # int
    num_rows,             # int
    num_cols,             # int
    row_stride,           # int
    col_stride,           # int,
    apply_smoothing,      # bool
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows

    # Load smoothing factor (scalar)
    smoothing = tl.load(smoothing_ptr)

    # dL/dx terms
    dloss = tl.load(dloss_ptr + offsets, mask=mask, other=0.0)
    row_lse = tl.load(lse_ptr + offsets, mask=mask, other=0.0)

    # We'll do step by step
    # For each row, compute p = exp(logits - lse)
    # Then grad = p * dloss
    # if label is i => gradient i = (p_i - 1) * dloss if no smoothing
    # with smoothing => we incorporate distribution

    col_range = tl.arange(0, BLOCK_SIZE)
    row_ptrs = logits_ptr + row_idx[:, None] * row_stride + col_range[None, :] * col_stride
    grad_ptrs = grads_ptr + row_idx[:, None] * row_stride + col_range[None, :] * col_stride

    for col_block_start in range(0, num_cols, BLOCK_SIZE):
        cols = col_block_start + col_range
        col_mask = cols < num_cols
        logits = tl.where(mask[:, None] & col_mask[None, :], tl.load(row_ptrs + col_range[None, :]), -float('inf'))
        logits = logits - row_lse[:, None]
        p = tl.exp(logits)
        # label smoothing modifies the "true label" distribution
        if apply_smoothing:
            p = p - (smoothing / num_cols)
        # standard cross entropy gradient
        # subtract 1.0 for the correct index
        label = tl.load(labels_ptr + offsets, mask=mask, other=-1)
        same_label = (label == cols) & (label != ignored_index)
        grad_val = tl.where(same_label, p - (1.0 - smoothing), p)
        # multiply by dloss
        grad_val = grad_val * dloss[:, None]
        # extra derivative for LSE square scale
        # derivative wrt logit => lse_square_scale * lse * (d(lse)/d(logit) + lse)
        # approximate: d(lse)/d(logit) = p
        # so total: lse_square_scale * row_lse * (lse * p + lse)
        grad_val += lse_square_scale * row_lse[:, None] * row_lse[:, None] * p * 0.5

        # store
        tl.store(grad_ptrs + col_range[None, :], tl.where(mask[:, None] & col_mask[None, :], grad_val, 0.0))

        row_ptrs += BLOCK_SIZE * col_stride
        grad_ptrs += BLOCK_SIZE * col_stride


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, lse_square_scale, ignored_index, process_group=None):
        # shapes
        num_rows, num_cols = logits.shape
        # allocate output
        loss = torch.empty_like(labels, dtype=logits.dtype)
        lse = torch.empty_like(labels, dtype=logits.dtype)
        smoothing_tensor = torch.tensor([smoothing], dtype=logits.dtype, device=logits.device)

        # launch forward kernel
        BLOCK_SIZE = 128
        grid = lambda meta: ( (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE, )
        cross_entropy_fwd_kernel[grid](
            logits,
            labels,
            loss,
            lse,
            smoothing_tensor,
            lse_square_scale,
            ignored_index,
            num_rows,
            num_cols,
            logits.stride(0),
            logits.stride(1),
            1 if smoothing > 0.0 else 0,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # sum the loss across all ranks if needed (tensor parallel / distributed)
        if process_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=process_group)

        ctx.save_for_backward(logits, labels, loss, lse, smoothing_tensor)
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.process_group = process_group
        return loss

    @staticmethod
    def backward(ctx, dloss):
        logits, labels, loss, lse, smoothing_tensor = ctx.saved_tensors
        num_rows, num_cols = logits.shape
        grads = torch.empty_like(logits)
        BLOCK_SIZE = 128
        grid = lambda meta: ((num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE, )
        cross_entropy_bwd_kernel[grid](
            logits,
            labels,
            dloss,
            lse,
            grads,
            smoothing_tensor,
            ctx.lse_square_scale,
            ctx.ignored_index,
            num_rows,
            num_cols,
            logits.stride(0),
            logits.stride(1),
            1 if smoothing_tensor.item() > 0.0 else 0,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # if distributed, no need to reduce grads since backward pass aggregator will do it
        return grads, None, None, None, None, None


def cross_entropy_loss(
    logits,
    labels,
    smoothing=0.0,
    lse_square_scale=0.0,
    ignored_index=-100,
    process_group=None
):
    return CrossEntropyLoss.apply(logits, labels, smoothing, lse_square_scale, ignored_index, process_group)
