import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr, COS_ptr, SIN_ptr, Y_ptr,
    BATCH, HEADS, SEQ, HEADDIM,
    IS_VARLEN, INTERLEAVED, CONJUGATE,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Calculate indices for batch, head, and sequence
    batch_id = pid // (HEADS * SEQ)
    head_id = (pid // SEQ) % HEADS
    seq_id = pid % SEQ

    # Base pointers for the current batch, head, and sequence
    X_offset = batch_id * HEADS * SEQ * HEADDIM + head_id * SEQ * HEADDIM + seq_id * HEADDIM
    COS_offset = head_id * HEADDIM
    SIN_offset = head_id * HEADDIM
    Y_offset = X_offset

    # Load data
    X = tl.load(X_ptr + X_offset + tl.arange(0, BLOCK_K))
    COS = tl.load(COS_ptr + COS_offset + tl.arange(0, BLOCK_K))
    SIN = tl.load(SIN_ptr + SIN_offset + tl.arange(0, BLOCK_K))

    # Apply rotary positional encoding
    if CONJUGATE:
        SIN = -SIN

    X_rotated_real = X * COS - X * SIN
    X_rotated_imag = X * SIN + X * COS

    # Store result
    tl.store(Y_ptr + Y_offset + tl.arange(0, BLOCK_K), X_rotated_real + X_rotated_imag)

import torch

def apply_rotary(X, COS, SIN, IS_VARLEN=False, INTERLEAVED=False, CONJUGATE=False):
    # Ensure tensor contiguity
    X = X.contiguous()
    COS = COS.contiguous()
    SIN = SIN.contiguous()

    # Extract dimensions
    BATCH, HEADS, SEQ, HEADDIM = X.shape

    # Prepare output tensor
    Y = torch.empty_like(X)

    # Define block and grid sizes
    BLOCK_M = 1  # Single sequence processing
    BLOCK_K = HEADDIM  # Process one head dimension at a time
    grid = (BATCH * HEADS * SEQ,)

    # Launch kernel
    rotary_kernel[grid](
        X, COS, SIN, Y,
        BATCH, HEADS, SEQ, HEADDIM,
        IS_VARLEN, INTERLEAVED, CONJUGATE,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )

    return Y
