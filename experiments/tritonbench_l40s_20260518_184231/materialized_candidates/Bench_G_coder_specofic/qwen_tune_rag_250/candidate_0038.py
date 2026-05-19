_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
pass

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride.to(tl.int64)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr).to(tl.float32)
    else:
        dloss = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    probs = tl.exp(logits - logsumexp)
    probs = probs.to(tl.float32)
    if label_idx != -100:
        probs += -dloss * tl.exp(logits[label_idx] - logsumexp)
    tl.store(logits_ptr + col_offsets, probs, mask=mask)
pass

@triton.jit
def _chunked_cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride.to(tl.int64)
    logsumexp = tl.load(logsumexp_ptr + row_idx * N_CHUNKS + chunk_idx)
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr).to(tl.float32)
    else:
        dloss = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    probs = tl.exp(logits - logsumexp)
    probs = probs.to(tl.float32)
    if label_idx != -100:
        probs += -dloss * tl.exp(logits[label_idx] - logsumexp)
    tl.store(logits_ptr + col_offsets, probs, mask=mask)
pass

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits, labels, ignore_index=-100,
        smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0,
        softcap=None, cap_temperature=1.0, max_cap_value=None,
    ):
        if softcap is not None:
            softcap *= cap_temperature
            if max_cap_value is not None:
                max_cap_value *= cap_temperature
        has_smoothing = smoothing > 0.0
        has_logit_scaling = logit_scale != 1.0
        has_lse_square = lse_square_scale > 0.0

        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        BLOCK_SIZE = max(1024, get_block_size(n_cols))
        N_CHUNKS = max(1, n_cols // BLOCK_SIZE)
        assert N_CHUNKS * BLOCK_SIZE >= n_cols

        logsumexp = torch.empty(n_rows if N_CHUNKS == 1 else n_rows * N_CHUNKS,
                                 dtype=torch.float32, device=logits.device)
        loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        if has_lse_square:
            z_loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        else:
            z_loss = None

        if N_CHUNKS == 1:
            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                loss,
                logsumexp,
                labels,
                n_cols, BLOCK_SIZE,
                DO_SOFTCAPPING=softcap is not None,
                SOFTCAP=softcap,
                DO_LOGIT_SCALING=has_logit_scaling,
                LOGIT_SCALE=logit_scale,
            )
            if has_lse_square:
                z_loss = 0.5 * lse_square_scale * (logsumexp ** 2)
        else:
            _chunked_cross_entropy_forward[(n_rows, N_CHUNKS,)](
                logits, logits.stride(0),
                loss,
                logsumexp,
                labels,
                n_cols, N_CHUNKS, BLOCK_SIZE,
                DO_SOFTCAPPING=softcap is not None,
                SOFTCAP=softcap,
                DO_LOGIT_SCALING=has_logit_scaling,
                LOGIT_SCALE=logit_scale,
            )
            if has_lse_square:
                z_loss = 0.5 * lse_square_scale * (logsumexp.reshape(n_rows, N_CHUNKS).sum(1))
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.softcap = softcap
        ctx.max_cap_value = max_cap_value
        ctx.logit_scale = logit_scale
        ctx.smoothing = smoothing
        ctx.has_logit_scaling = has_logit_scaling
        ctx.has_smoothing = has_smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        ctx.N_CHUNKS = N_CHUNKS
        return loss, z_loss if has_lse_square else None

    @staticmethod
    def backward(ctx, dloss, ddummy=None):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = max(1024, get_block_size(n_cols))
        N_CHUNKS = max(1, n_cols // BLOCK_SIZE)
        assert N_CHUNKS * BLOCK_SIZE >= n_cols

        if N_CHUNKS == 1:
            _cross_entropy_backward[(n_rows,)](
                logits, logits.stride(0),
                dloss, dloss.stride(0),
                logsumexp,
                labels,
                n_cols, BLOCK_SIZE,
                DO_SOFTCAPPING=ctx.softcap is not None,
                SOFTCAP=ctx.softcap,
                DO_LOGIT_SCALING=ctx.has_logit_scaling,
                LOGIT_SCALE=ctx.logit_scale,
            )
        else:
            _chunked_cross_entropy_backward[(n_rows, N_CHUNKS,)](
                logits, logits.stride(0),
                dloss, dloss.stride(0),
                logsumexp,
                labels,
                n_cols, N_CHUNKS, BLOCK_SIZE,
                DO_SOFTCAPPING=ctx.softcap is not None,
                SOFTCAP=ctx.softcap,
                DO_LOGIT_SCALING=ctx.has_logit_scaling,
                LOGIT_SCALE=ctx.logit_scale,
            )
        return logits, None, None, None, None, None, None, None, None,
pass

def cross_entropy(
    input, target, ignore_index=-100,
    smoothing=0.0, logit_scale=1
