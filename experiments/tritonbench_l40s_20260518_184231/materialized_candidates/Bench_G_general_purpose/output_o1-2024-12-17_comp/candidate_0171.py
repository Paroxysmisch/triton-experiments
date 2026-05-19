import math
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 16384

def calculate_settings(n: int):
    block_size = 1 << ((n - 1).bit_length())  # next power of 2 >= n
    block_size = min(block_size, MAX_FUSED_SIZE)
    if block_size >= 1024:
        num_warps = 8
    elif block_size >= 512:
        num_warps = 4
    else:
        num_warps = 2
    return block_size, num_warps

@triton.jit
def _cross_entropy_forward(
    logits_ptr, labels_ptr, losses_ptr,
    vocab_size, row_stride, has_label, softcap, log_scaling,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    off = row_idx * row_stride
    block_offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < (off + vocab_size)
    logits = tl.load(logits_ptr + block_offsets, mask=mask, other=-1e30)

    if softcap > 0.0:
        logits = tl.maximum(logits, softcap)
    if log_scaling != 1.0:
        logits = logits * log_scaling

    # Compute max for log-sum-exp stability
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = max_logit + tl.log(sum_exp)

    # Store partial result in losses
    # If we have valid label, compute final loss
    if has_label:
        label_val = tl.load(labels_ptr + row_idx)
        if label_val != -100:
            label_mask = (tl.arange(0, BLOCK_SIZE) == label_val)
            logit_label = tl.sum(logits * label_mask, axis=0)
            loss_val = log_sum_exp - logit_label
        else:
            loss_val = 0.0
        tl.store(losses_ptr + row_idx, loss_val)
    else:
        # For chest or partial usage
        tl.store(losses_ptr + row_idx, log_sum_exp)

@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, labels_ptr, partial_sums_ptr,
    vocab_size, row_stride, chunk_offset, chunk_size,
    row_idx, has_label, softcap, log_scaling,
    BLOCK_SIZE: tl.constexpr
):
    off = row_idx * row_stride + chunk_offset
    block_offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < (row_idx * row_stride + vocab_size)
    logits = tl.load(logits_ptr + block_offsets, mask=mask, other=-1e30)

    if softcap > 0.0:
        logits = tl.maximum(logits, softcap)
    if log_scaling != 1.0:
        logits = logits * log_scaling

    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)

    # Write partial sums
    # partial_sums_ptr layout: [row_idx * 2], gather max and sum
    tl.store(partial_sums_ptr + row_idx * 2 + 0, max_logit, mask=True)
    tl.store(partial_sums_ptr + row_idx * 2 + 1, sum_exp, mask=True)

@triton.jit
def _cross_entropy_backward(
    logits_ptr, labels_ptr, dlogits_ptr, dlosses_ptr,
    vocab_size, row_stride, softcap, log_scaling,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    off = row_idx * row_stride
    block_offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < (off + vocab_size)
    logits = tl.load(logits_ptr + block_offsets, mask=mask, other=-1e30)
    dloss = tl.load(dlosses_ptr + row_idx)

    if softcap > 0.0:
        logits = tl.maximum(logits, softcap)
    if log_scaling != 1.0:
        logits = logits * log_scaling

    # softmax
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sum_exp

    label_val = tl.load(labels_ptr + row_idx)
    grad = softmax
    if label_val != -100:
        label_mask = (tl.arange(0, BLOCK_SIZE) == label_val)
        grad = grad - label_mask

    if log_scaling != 1.0:
        grad = grad * log_scaling
    tl.store(dlogits_ptr + block_offsets, grad * dloss, mask=mask)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, softcap=0.0, log_scaling=1.0):
        B, N = logits.shape
        block_size, num_warps = calculate_settings(N)
        out = torch.empty((B,), device=logits.device, dtype=logits.dtype)

        if N <= MAX_FUSED_SIZE:
            grid = (B,)
            _cross_entropy_forward[grid](
                logits, labels, out,
                N, N, True, softcap, log_scaling,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            partial_log_sum = torch.zeros((B, 2), device=logits.device, dtype=logits.dtype)
            chunk_size = block_size
            chunks = (N + chunk_size - 1) // chunk_size
            for i in range(chunks):
                chunk_offset = i * chunk_size
                grid = (B,)
                _chunked_cross_entropy_forward[grid](
                    logits, labels, partial_log_sum,
                    N, N, chunk_offset, chunk_size,
                    tl.arange(0, B), True, softcap, log_scaling,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )

            # Combine partial results
            # partial sums in partial_log_sum: each row has [max_logit, sum_exp]
            # We must do a stable combine of max and sums
            # This is a naive CPU side example:
            max_vals = partial_log_sum[:, 0].reshape(B, -1)
            sum_vals = partial_log_sum[:, 1].reshape(B, -1)
            # sum_vals only has 1 column since we store row by row: shape=(B,1)
            combined = max_vals + torch.log(sum_vals)
            # If needed, replicate partial approach or do row by row
            out[:] = combined[:, 0]

            # Subtract label logit for each row
            valid_mask = labels != -100
            idx = labels[valid_mask].long()
            row_idxs = torch.nonzero(valid_mask).flatten()
            gather_vals = logits[row_idxs, idx]
            if softcap > 0.0:
                gather_vals = torch.maximum(gather_vals, torch.tensor(softcap, device=gather_vals.device, dtype=gather_vals.dtype))
            if log_scaling != 1.0:
                gather_vals = gather_vals * log_scaling
            out[row_idxs] = out[row_idxs] - gather_vals

        ctx.save_for_backward(logits, labels, torch.tensor(softcap, device=logits.device, dtype=logits.dtype),
                              torch.tensor(log_scaling, device=logits.device, dtype=logits.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, softcap, log_scaling = ctx.saved_tensors
        B, N = logits.shape
        block_size, num_warps = calculate_settings(N)
        dlogits = torch.empty_like(logits)

        grid = (B,)
        _cross_entropy_backward[grid](
            logits, labels, dlogits, grad_output,
            N, N, softcap.item(), log_scaling.item(),
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        return dlogits, None, None, None

def fast_cross_entropy_loss(logits, labels, softcap=0.0, log_scaling=1.0):
    logits_2d = logits.view(-1, logits.shape[-1])
    labels_1d = labels.view(-1)
    losses = Fast_CrossEntropyLoss.apply(logits_2d, labels_1d, softcap, log_scaling)
    # Mask out -100
    valid_mask = labels_1d != -100
    mean_loss = (losses[valid_mask].sum() / (valid_mask.sum() + 1e-12))
    return mean_loss, losses.view(labels.shape)
