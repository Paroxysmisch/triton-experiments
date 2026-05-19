import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 4096

def calculate_settings(n):
    """Determine block size and number of warps."""
    if n <= MAX_FUSED_SIZE // 2:
        return MAX_FUSED_SIZE, 8
    elif n <= MAX_FUSED_SIZE:
        return MAX_FUSED_SIZE, 4
    else:
        return MAX_FUSED_SIZE, 2

@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    labels_ptr,
    losses_ptr, log_sum_exp_ptr,
    n_valid_label_ptr,
    cap_ptr,
    logit_scale_ptr,
    ignore_index,
    n: tl.constexpr, d: tl.constexpr, n_warps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
    Softmax(Pi) = exp(xi) / sum(exp(xi))
    CE = - sum( y * log(x) )
    y = 1(n[i] == c) -> 1-hot encoding
    x = exp(xi) / sum(exp(xi))
    d/dx CE = -1/n * [ sum( y * 1 ) - sum( y * log(x) * 1/n * exp(xi) / sum(exp(x)) ) ]
    d/dx CE = -1/n * [ sum( y * 1 ) - sum( y * log(x) * exp(xi) / sum(exp(x)) ) ]
    d/dx CE = -1/n * [ sum( y * 1 ) - sum( y * log(x * exp(xi) / sum(exp(x))) ) ]
    d/dx CE = -1/n * [ sum( y * 1 ) - sum( y * log(x) * exp(xi) / sum(exp(x)) ) ]
    """
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    labels_ptr = labels_ptr + row_idx
    losses_ptr = losses_ptr + row_idx
    log_sum_exp_ptr = log_sum_exp_ptr + row_idx
    cap_ptr = cap_ptr + row_idx
    logit_scale_ptr = logit_scale_ptr + row_idx

    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    logits = tl.where(tl.arange(0, BLOCK_SIZE) < d, logits, 0.0)
    label = tl.load(labels_ptr)
    n_valid_label = 0

    if label != ignore_index:
        cap = tl.load(cap_ptr)
        logit_scale = tl.load(logit_scale_ptr)
        logits = logits * logit_scale + cap
        logits = logits.to(tl.float32)
        if label >= 0:
            logits = logits + cap
            c = tl.load(logits_ptr + label)
            c = tl.exp(c)
            logits = logits - c
            c = tl.exp(c)
        else:
            logits = logits + cap

        log_sum_exp = tl.max(logits, 0)
        logits = tl.exp(logits - log_sum_exp)
        n_valid_label = 1

    loss = log_sum_exp - tl.log(n_valid_label + tl.sum(logits))
    tl.store(losses_ptr, loss)
    tl.store(log_sum_exp_ptr, log_sum_exp)
    tl.store(n_valid_label_ptr + row_idx, n_valid_label)

@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    labels_ptr,
    losses_ptr, log_sum_exp_ptr,
    n_valid_label_ptr,
    cap_ptr,
    logit_scale_ptr,
    ignore_index,
    n: tl.constexpr, d: tl.constexpr, n_warps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, N_CHUNKS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    labels_ptr = labels_ptr + row_idx
    losses_ptr = losses_ptr + row_idx
    log_sum_exp_ptr = log_sum_exp_ptr + row_idx
    cap_ptr = cap_ptr + row_idx
    logit_scale_ptr = logit_scale_ptr + row_idx

    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE) + chunk_idx * BLOCK_SIZE).to(tl.float32)
    logits = tl.where((tl.arange(0, BLOCK_SIZE) + chunk_idx * BLOCK_SIZE) < d, logits, 0.0)
    label = tl.load(labels_ptr)
    n_valid_label = 0

    if label != ignore_index:
        cap = tl.load(cap_ptr)
        logit_scale = tl.load(logit_scale_ptr)
        logits = logits * logit_scale + cap
        logits = logits.to(tl.float32)
        if label >= 0:
            logits = logits + cap
            c = tl.load(logits_ptr + label)
            c = tl.exp(c)
            logits = logits - c
            c = tl.exp(c)
        else:
            logits = logits + cap

        log_sum_exp = tl.max(logits, 0)
        logits = tl.exp(logits - log_sum_exp)
        n_valid_label = 1

    loss = log_sum_exp - tl.log(n_valid_label + tl.sum(logits))
    tl.store(losses_ptr, loss)
    tl.store(log_sum_exp_ptr, log_sum_exp)
    tl.store(n_valid_label_ptr + row_idx, n_valid_label)

@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dlosses_ptr, dlosses_row_stride,
    cap_ptr,
    logit_scale_ptr,
    n: tl.constexpr, d: tl.constexpr, n_warps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlosses_ptr = dlosses_ptr + row_idx * dlosses_row_stride.to(tl.int64)
    cap_ptr = cap_ptr + row_idx
    logit_scale_ptr = logit_scale_ptr + row_idx

    cap = tl.load(cap_ptr)
    logit_scale = tl.load(logit_scale_ptr)
    dloss = tl.load(dlosses_ptr)
    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    logits = tl.where(tl.arange(0, BLOCK_SIZE) < d, logits, -1e5)
    logits = (logits + cap) * logit_scale
    x = tl.exp(logits)
    softmax = x / tl.sum(x)
    softmax = softmax.to(tl.float32)
    logits = logits.to(tl.float32)
    dloss = dloss.to(tl.float32)
    dx = (softmax - tl.exp(logits)) * dloss
    tl.store(logits_ptr + tl.arange(0, BLOCK_SIZE), dx)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, n_valid_label,
                cap, logit_scale, ignore_index):
        ctx.save_for_backward(logits, labels, cap, logit_scale)
        ctx.logit_scale = logit_scale
        ctx.cap = cap
        ctx.ignore_index = ignore_index

        n, d = logits.shape
        d = logits.size(dim=-1)
        losses = torch.empty(n, dtype=torch.float, device=logits.device)
        log_sum_exp = torch.empty(n, dtype=torch.float, device=logits.device)
        n_valid_label = torch.empty(n, dtype=torch.float, device=logits.device)

        BLOCK_SIZE, n_warps = calculate_settings(d)

        if d > MAX_FUSED_SIZE:
            N_CHUNKS = d // MAX_FUSED_SIZE
            d = MAX_FUSED_SIZE
            d_last = logits.size(dim=-1) % MAX_FUSED_SIZE
            _chunked_cross_entropy_forward[(n, N_CHUNKS)](
                logits, logits.stride(0),
                labels,
                losses, log_sum_exp, n_valid_label,
                cap, logit_scale,
                ignore_index,
                n, d, n_warps,
                BLOCK_SIZE,
                num_warps=n_warps,
                num_stages=1,
                N_CHUNKS=N_CHUNKS
            )
        else:
            _cross_entropy_forward[(n,)](
                logits, logits.stride(0),
                labels,
                losses, log_sum_exp, n_valid_label,
                cap, logit_scale,
                ignore_index,
                n, d, n_warps,
                BLOCK_SIZE,
                num_warps=n_warps,
                num_stages=1
            )

        n_valid_label = n_valid_label.to(torch.int)
        losses = losses * n_valid_label
        mean_loss = losses.sum() / n_valid_label.sum()
        ctx.mean_loss = mean_loss
        return mean_loss

    @staticmethod
    def backward(ctx, dlosses):
        logits, labels, cap, logit_scale = ctx.saved_tensors
        n, d = logits.shape

        BLOCK_SIZE, n_warps = calculate_settings(d)

        _cross_entropy_backward[(n,)](
            logits, logits.stride(0),
            dlosses, dlosses.stride(0),
            cap, logit_scale,
            n, d, n_warps,
            BLOCK_SIZE,
            num_warps=n_warps,
            num_st
