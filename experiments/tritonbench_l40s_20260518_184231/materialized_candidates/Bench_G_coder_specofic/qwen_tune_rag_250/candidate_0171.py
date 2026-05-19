= tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = logsumexp - x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    pass
    tl.store(logsumexp_ptr, logsumexp)
pass

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dlosses_ptr, dlosses_row_stride,
    logsumexp_ptr,
    labels_ptr,
    logits_grad_ptr, logits_grad_row_stride,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    k = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dlosses_ptr += row_idx * dlosses_row_stride.to(tl.int64)
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    logits_grad_ptr += row_idx * logits_grad_row_stride.to(tl.int64)

    col_offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr).to(tl.int32)
    dloss = tl.load(dlosses_ptr)
    logsumexp = tl.load(logsumexp_ptr)

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x *= LOGIT_SCALE
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        x = tl.exp(x.to(tl.float32) - logsumexp)
    else:
        x = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    if DO_LOGIT_SCALING: logits *= LOGIT_SCALE
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    logits = logits.to(tl.float32)
    softmax_out = tl.exp(logits - logsumexp)
    y = softmax_out / (1 + x)

    tl.store(logits_grad_ptr + col_offsets, y, mask=mask)
pass

class Fast_CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100, softcap=-1, logit_scale=1.0):
        use_chunks, n_chunks, VOCAB_SIZE, BLOCK_SIZE, _, _ = calculate_settings(logits)
        device = logits.device
        if use_chunks:
            losses = torch.empty((logits.size(0), ), dtype=torch.float32, device=device)
            logsumexp = torch.empty((logits.size(0), n_chunks), dtype=torch.float32, device=device)
            _chunked_cross_entropy_forward[(logits.size(0), n_chunks)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE,
                n_chunks,
                BLOCK_SIZE,
                DO_SOFTCAPPING=softcap > 0,
                SOFTCAP=softcap,
                DO_LOGIT_SCALING=logit_scale != 1.0,
                LOGIT_SCALE=logit_scale,
            )
            losses += logsumexp.sum(dim=1)
        else:
            losses = torch.empty((logits.size(0), ), dtype=torch.float32, device=device)
            logsumexp = torch.empty((logits.size(0), ), dtype=torch.float32, device=device)
            _cross_entropy_forward[(logits.size(0),)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE,
                BLOCK_SIZE,
                DO_SOFTCAPPING=softcap > 0,
                SOFTCAP=softcap,
                DO_LOGIT_SCALING=logit_scale != 1.0,
                LOGIT_SCALE=logit_scale,
            )
        if ignore_index != -100:
            losses[labels == ignore_index] = 0.0
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses.mean(), logits, logsumexp

    @staticmethod
    def backward(ctx, dloss, _logits, logsumexp):
        logits, _, labels = ctx.saved_tensors
        use_chunks, n_chunks, _, BLOCK_SIZE, _, _ = calculate_settings(logits)
        dlosses = dloss / logits.size(0)
        if use_chunks:
            dlogits = torch.empty((logits.size(0), logits.size(1)), dtype=torch.float32, device=logits.device)
            _chunked_cross_entropy_backward[(logits.size(0), n_chunks)](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp,
                labels,
                dlogits, dlogits.stride(0),
                VOCAB_SIZE=logits.size(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            dlogits = torch.empty((logits.size(0), logits.size(1)), dtype=torch.float32, device=logits.device)
            _cross_entropy_backward[(logits.size(0),)](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp,
                labels,
                dlogits, dlogits.stride(0),
                VOCAB_SIZE=logits.size(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return dlogits, None, None, None, None
pass

def fast_cross_entropy_loss(logits, labels, ignore_index=-100, softcap=-1, logit_scale=1.0, perform_divide=True):
    use_chunks, _, _, _, N_ROWS, N_COLS = calculate_settings(logits)
    if perform_divide:
        return Fast_CrossEntropyLoss.apply(logits, labels, ignore_index, softcap, logit_scale)[0] / N_ROWS
    else:
        return Fast_CrossEntropyLoss.apply(logits, labels, ignore_index, softcap, logit_scale)[0]
pass
