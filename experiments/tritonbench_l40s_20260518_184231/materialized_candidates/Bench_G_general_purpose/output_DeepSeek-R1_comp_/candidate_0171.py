import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 4096

def calculate_settings(n):
    block_size = min(n, MAX_FUSED_SIZE)
    num_warps = (block_size + 31) // 32
    return {"block_size": block_size, "num_warps": num_warps}

@triton.jit
def _cross_entropy_forward(
    logits_ptr, labels_ptr, losses_ptr, log_sum_exp_ptr,
    batch_size, n_cols,
    use_softcap: tl.constexpr, softcap_scale: tl.constexpr,
    logit_scale: tl.constexpr,
    LOGITS_STRIDE: tl.constexpr, LOSSES_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    label = tl.load(labels_ptr + row_idx)
    valid = label != -100
    
    row_start = row_idx * LOGITS_STRIDE
    max_logit = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for col_off in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_off + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        logits = tl.load(logits_ptr + row_start + col_idx, mask=mask, other=0)

        if use_softcap:
            logits = tl.tanh(logits) * softcap_scale
        if logit_scale != 1.0:
            logits = logits * logit_scale

        curr_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, curr_max)

    max_logit = tl.max(max_logit, axis=0)

    for col_off in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_off + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        logits = tl.load(logits_ptr + row_start + col_idx, mask=mask, other=0)

        if use_softcap:
            logits = tl.tanh(logits) * softcap_scale
        if logit_scale != 1.0:
            logits = logits * logit_scale

        shifted = logits - max_logit
        exp = tl.exp(shifted)
        sum_exp += tl.sum(exp, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_logit
    tl.store(log_sum_exp_ptr + row_idx, log_sum_exp)

    if valid:
        target = tl.load(logits_ptr + row_start + label)
        if use_softcap:
            target = tl.tanh(target) * softcap_scale
        if logit_scale != 1.0:
            target = target * logit_scale
        loss = log_sum_exp - target
    else:
        loss = 0.0

    tl.store(losses_ptr + row_idx * LOSSES_STRIDE, loss)

@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, labels_ptr, losses_ptr, log_sum_exp_ptr,
    batch_size, n_cols,
    use_softcap: tl.constexpr, softcap_scale: tl.constexpr,
    logit_scale: tl.constexpr,
    LOGITS_STRIDE: tl.constexpr, LOSSES_STRIDE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    label = tl.load(labels_ptr + row_idx)
    valid = label != -100
    row_start = row_idx * LOGITS_STRIDE

    global_max = -float('inf')
    total_sum = 0.0
    target_logit = 0.0

    num_chunks = tl.cdiv(n_cols, CHUNK_SIZE)
    for chunk in range(num_chunks):
        start = chunk * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_cols)
        cols = start + tl.arange(0, CHUNK_SIZE)
        mask = cols < end
        
        logits = tl.load(logits_ptr + row_start + cols, mask=mask, other=0)
        if use_softcap:
            logits = tl.tanh(logits) * softcap_scale
        if logit_scale != 1.0:
            logits = logits * logit_scale

        chunk_max = tl.max(logits, axis=0)
        new_max = tl.maximum(global_max, chunk_max)
        if chunk > 0:
            total_sum *= tl.exp(global_max - new_max)
        
        shifted = logits - new_max
        exp_vals = tl.exp(shifted)
        chunk_sum = tl.sum(exp_vals, axis=0)
        total_sum += chunk_sum
        global_max = new_max

        if valid & (start <= label) & (label < end):
            pos = label - start
            target_logit = tl.load(logits_ptr + row_start + start + pos)
            if use_softcap:
                target_logit = tl.tanh(target_logit) * softcap_scale
            if logit_scale != 1.0:
                target_logit *= logit_scale

    log_sum_exp = tl.log(total_sum) + global_max
    tl.store(log_sum_exp_ptr + row_idx, log_sum_exp)
    loss = tl.where(valid, log_sum_exp - target_logit, 0.0)
    tl.store(losses_ptr + row_idx * LOSSES_STRIDE, loss)

@triton.jit
def _cross_entropy_backward(
    dlosses_ptr, logits_ptr, labels_ptr, log_sum_exp_ptr, dlogits_ptr,
    batch_size, n_cols,
    use_softcap: tl.constexpr, softcap_scale: tl.constexpr,
    logit_scale: tl.constexpr,
    DLOGITS_STRIDE: tl.constexpr, LOGITS_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    label = tl.load(labels_ptr + row_idx)
    valid = label != -100
    dloss = tl.load(dlosses_ptr + row_idx)
    log_sum_exp = tl.load(log_sum_exp_ptr + row_idx)
    row_start = row_idx * LOGITS_STRIDE

    for col_off in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_off + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        logits = tl.load(logits_ptr + row_start + col_idx, mask=mask, other=0)
        if use_softcap:
            capped = tl.tanh(logits) * softcap_scale
        else:
            capped = logits * logit_scale
        
        probs = tl.exp(capped - log_sum_exp)
        grad = probs * dloss

        if valid and (col_off <= label < col_off + BLOCK_SIZE):
            pos = label - col_off
            on_target = tl.where(tl.arange(0, BLOCK_SIZE) == pos, 1.0, 0.0)
            grad -= on_target * dloss

        if use_softcap:
            deriv = softcap_scale * (1 - tl.tanh(logits)**2)
            grad *= deriv
        if logit_scale != 1.0:
            grad *= logit_scale

        tl.store(dlogits_ptr + row_idx * DLOGITS_STRIDE + col_idx, grad, mask=mask)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, use_softcap=False, softcap_scale=1.0, logit_scale=1.0):
        batch_size, n_cols = logits.shape
        losses = torch.empty_like(logits[:, 0])
        log_sum_exp = torch.empty_like(logits[:, 0])
        
        if n_cols <= MAX_FUSED_SIZE:
            settings = calculate_settings(n_cols)
            _cross_entropy_forward[(batch_size,)](
                logits, labels, losses, log_sum_exp,
                batch_size, n_cols,
                use_softcap, softcap_scale, logit_scale,
                logits.stride(0), losses.stride(0),
                **settings
            )
        else:
            _chunked_cross_entropy_forward[(batch_size,)](
                logits, labels, losses, log_sum_exp,
                batch_size, n_cols,
                use_softcap, softcap_scale, logit_scale,
                logits.stride(0), losses.stride(0),
                MAX_FUSED_SIZE
            )
        
        ctx.save_for_backward(logits, labels, log_sum_exp)
        ctx.use_softcap = use_softcap
        ctx.softcap_scale = softcap_scale
        ctx.logit_scale = logit_scale
        
        valid = labels != -100
        return losses[valid].mean() if valid.any() else losses.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, log_sum_exp = ctx.saved_tensors
        dlogits = torch.zeros_like(logits)
        batch_size = logits.size(0)
        
        settings = calculate_settings(logits.size(1))
        _cross_entropy_backward[(batch_size,)](
            grad_output, logits, labels, log_sum_exp, dlogits,
            batch_size, logits.size(1),
            ctx.use_softcap, ctx.softcap_scale, ctx.logit_scale,
            dlogits.stride(0), logits.stride(0),
            **settings
        )
        
        valid = labels != -100
        dlogits[valid] *= grad_output / valid.float().sum()
        return dlogits, None, None, None, None

def fast_cross_entropy_loss(logits, labels, use_softcap=False, softcap_scale=1.0, logit_scale=1.0):
    return Fast_CrossEntropyLoss.apply(logits, labels, use_softcap, softcap_scale, logit_scale)
