import triton
import triton.language as tl
import torch
from .utils import triton_tanh

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,  # Input logits pointer and stride
    loss_ptr,                       # Output loss pointer 
    logsumexp_ptr,                  # Output logsumexp pointer
    labels_ptr,                     # Input labels pointer
    VOCAB_SIZE: tl.constexpr,       # Size of vocabulary
    BLOCK_SIZE: tl.constexpr,       # Processing block size
    DO_SOFTCAPPING: tl.constexpr,   # Whether to apply softcapping
    SOFTCAP: tl.constexpr,          # Softcapping value
    DO_LOGIT_SCALING: tl.constexpr, # Whether to scale logits
    LOGIT_SCALE: tl.constexpr,      # Logit scaling factor
):
    # Get row index and calculate pointer offsets
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    # Create column offsets and mask for valid vocab indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    # Load label and logits
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Apply optional logit scaling and softcapping
    if DO_LOGIT_SCALING: 
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: 
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    # Compute logsumexp using the log-sum-exp trick for numerical stability
    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    # Compute loss if label is valid (-100 is ignore index)
    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: 
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: 
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0

    # Store results
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
