def calculate_settings(n):
    """Calculate optimal block size and number of warps based on vocabulary size."""
    if n <= 512:
        block_size = 512
        num_warps = 4
    elif n <= 1024:
        block_size = 1024
        num_warps = 8
    elif n <= 2048:
        block_size = 2048
        num_warps = 16
    else:
        block_size = min(4096, n)
        num_warps = 32
    return block_size, num_warps

// ... existing code ...

@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr, 
    LOGIT_SCALE: tl.constexpr,
):
    # Get row index and calculate pointer offsets
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    
    # Load label and check if valid
    label_idx = tl.load(labels_ptr + row_idx)
    
    # Load and transform logits
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    logits = tl.load(logits_ptr + col_offsets, mask=mask)
    
    # Apply optional transformations
    if DO_LOGIT_SCALING:
        logits = logits * LOGIT_SCALE
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
        
    # Calculate logsumexp for numerical stability
    max_logit = tl.max(logits, 0) 
    exp_logits = tl.exp(logits - max_logit)
    logsumexp = max_logit + tl.log(tl.sum(exp_logits, 0))
    
    # Calculate loss if label is valid
    loss = tl.where(
        label_idx != -100,
        logsumexp - logits[label_idx],
        0.0
    )
    
    # Store results
    tl.store(loss_ptr + row_idx, loss)
    tl.store(logsumexp_ptr + row_idx, logsumexp)

// ... existing code ...

@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    grad_ptr, grad_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    # Calculate indices and offsets
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Load gradient and label
    grad = tl.load(grad_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx)
    
    # Calculate softmax gradients
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    
    logits = tl.load(logits_ptr + col_offsets, mask=mask)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    
    # Apply transformations and calculate gradients
    if DO_LOGIT_SCALING:
        logits = logits * LOGIT_SCALE
    if DO_SOFTCAPPING:
        tanh_term = triton_tanh(logits / SOFTCAP)
        logits = SOFTCAP * tanh_term
        
    probs = tl.exp(logits - logsumexp)
    grad_output = tl.where(
        col_offsets == label_idx,
        probs - 1.0,
        probs
    )
    
    # Apply chain rule for transformations
    if DO_LOGIT_SCALING:
        grad_output = grad_output * LOGIT_SCALE
    if DO_SOFTCAPPING:
        grad_output = grad_output * (1.0 - tanh_term * tanh_term)
        
    # Store gradients
    tl.store(
        logits_ptr + col_offsets,
        grad * grad_output,
        mask=mask
    )

// ... existing code ...
