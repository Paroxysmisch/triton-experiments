import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 2048  # Maximum fused size for a single block

# Function to determine optimal block size and number of warps
def calculate_settings(n):
    block_size = min(n, MAX_FUSED_SIZE)
    num_warps = (block_size + 31) // 32  # Ensure at least one warp
    return block_size, num_warps

# Forward kernel for cross-entropy loss
@triton.jit
def _cross_entropy_forward(
    logits_ptr, labels_ptr, losses_ptr, n, block_size: tl.constexpr,
    apply_softcap: tl.constexpr, logit_scale: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n

    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    labels = tl.load(labels_ptr + pid, mask=mask, other=-100)

    if apply_softcap:
        logits = tl.log1p(tl.exp(logits))  # Softcap

    if logit_scale != 1.0:
        logits = logits * logit_scale  # Logit scaling

    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp_logits) + max_logit

    if labels != -100:
        true_logit = tl.load(logits_ptr + labels, mask=mask, other=0.0)
        loss = log_sum_exp - true_logit
    else:
        loss = 0.0

    tl.store(losses_ptr + pid, loss, mask=mask)

# Chunked forward kernel for large vocabularies
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, labels_ptr, losses_ptr, n, block_size: tl.constexpr,
    apply_softcap: tl.constexpr, logit_scale: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_chunks = (n + block_size - 1) // block_size
    log_sum_exp = -float('inf')

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * block_size
        chunk_end = min(chunk_start + block_size, n)
        offsets = chunk_start + tl.arange(0, block_size)
        mask = (offsets >= chunk_start) & (offsets < chunk_end)

        chunk_logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))

        if apply_softcap:
            chunk_logits = tl.log1p(tl.exp(chunk_logits))  # Softcap

        if logit_scale != 1.0:
            chunk_logits = chunk_logits * logit_scale  # Logit scaling

        max_logit = tl.max(chunk_logits, axis=0)
        chunk_logits = chunk_logits - max_logit
        exp_chunk_logits = tl.exp(chunk_logits)
        sum_exp_chunk_logits = tl.sum(exp_chunk_logits, axis=0)
        log_sum_exp_chunk = tl.log(sum_exp_chunk_logits) + max_logit

        log_sum_exp = tl.max(log_sum_exp, log_sum_exp_chunk)

    labels = tl.load(labels_ptr + pid, mask=mask, other=-100)

    if labels != -100:
        true_logit = tl.load(logits_ptr + labels, mask=mask, other=0.0)
        loss = log_sum_exp - true_logit
    else:
        loss = 0.0

    tl.store(losses_ptr + pid, loss, mask=mask)

# Backward kernel for cross-entropy loss
@triton.jit
def _cross_entropy_backward(
    logits_ptr, labels_ptr, dlosses_ptr, dlogits_ptr, n, block_size: tl.constexpr,
    apply_softcap: tl.constexpr, logit_scale: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n

    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    labels = tl.load(labels_ptr + pid, mask=mask, other=-100)
    dlosses = tl.load(dlosses_ptr + pid, mask=mask, other=0.0)

    if apply_softcap:
        logits = tl.log1p(tl.exp(logits))  # Softcap

    if logit_scale != 1.0:
        logits = logits * logit_scale  # Logit scaling

    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sum_exp_logits

    if labels != -100:
        true_logit = tl.load(logits_ptr + labels, mask=mask, other=0.0)
        grad = softmax - (true_logit == logits)
        grad = grad * dlosses
    else:
        grad = 0.0

    tl.store(dlogits_ptr + offsets, grad, mask=mask)

# Fast_CrossEntropyLoss class
class Fast_CrossEntropyLoss:
    def __init__(self, apply_softcap=False, logit_scale=1.0):
        self.apply_softcap = apply_softcap
        self.logit_scale = logit_scale

    def forward(self, logits, labels):
        n = logits.shape[1]
        block_size, num_warps = calculate_settings(n)
        losses = triton.empty_like(logits)

        if n <= MAX_FUSED_SIZE:
            _cross_entropy_forward[logits.shape[0], num_warps](logits, labels, losses, n, block_size, self.apply_softcap, self.logit_scale)
        else:
            _chunked_cross_entropy_forward[logits.shape[0], num_warps](logits, labels, losses, n, block_size, self.apply_softcap, self.logit_scale)

        return losses

    def backward(self, logits, labels, dlosses):
        n = logits.shape[1]
        block_size, num_warps = calculate_settings(n)
        dlogits = triton.empty_like(logits)

        _cross_entropy_backward[logits.shape[0], num_warps](logits, labels, dlosses, dlogits, n, block_size, self.apply_softcap, self.logit_scale)

        return dlogits

# Function to process batched inputs and compute mean loss
def fast_cross_entropy_loss(logits, labels, apply_softcap=False, logit_scale=1.0):
    loss_fn = Fast_CrossEntropyLoss(apply_softcap, logit_scale)
    losses = loss_fn.forward(logits, labels)
    valid_mask = labels != -100
    valid_losses = losses[valid_mask]
    mean_loss = triton.sum(valid_losses) / triton.sum(valid_mask)
    return mean_loss
