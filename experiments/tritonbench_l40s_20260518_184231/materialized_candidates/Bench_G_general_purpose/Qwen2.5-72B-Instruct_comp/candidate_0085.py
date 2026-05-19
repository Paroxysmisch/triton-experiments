import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Pointer to sequence lengths for batches
    Mid_O,     # Pointer to input tensor [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # Pointer to input tensor [batch, head, seq_block_num]
    O,         # Pointer to output tensor [batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,  # Strides for Mid_O
    stride_logexpsum_ob, stride_logexpsum_oh, stride_logexpsum_os,  # Strides for Mid_O_LogExpSum
    stride_o_ob, stride_o_oh, stride_o_od,  # Strides for O
    BLOCK_SEQ: tl.constexpr,  # Block size for sequences
    BLOCK_DMODEL: tl.constexpr  # Block size for model dimensions
):
    # Get program ID
    pid = tl.program_id(axis=0)
    b = pid // (BLOCK_SEQ * BLOCK_DMODEL)
    h = (pid % (BLOCK_SEQ * BLOCK_DMODEL)) // BLOCK_DMODEL
    d = (pid % (BLOCK_SEQ * BLOCK_DMODEL)) % BLOCK_DMODEL

    # Compute the sequence length for this batch
    seqlen = tl.load(B_Seqlen + b)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    logexpsum = tl.zeros((1,), dtype=tl.float32)

    # Iterate over sequence blocks
    for s in range(0, seqlen, BLOCK_SEQ):
        # Load Mid_O and Mid_O_LogExpSum
        mid_o = tl.load(Mid_O + b * stride_mid_ob + h * stride_mid_oh + s * stride_mid_os + d * stride_mid_od, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)
        mid_o_logexpsum = tl.load(Mid_O_LogExpSum + b * stride_logexpsum_ob + h * stride_logexpsum_oh + s * stride_logexpsum_os, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)

        # Compute new values
        new_acc = mid_o * tl.exp(mid_o_logexpsum - logexpsum)
        logexpsum = tl.log(tl.exp(logexpsum) + tl.exp(mid_o_logexpsum))

        # Update accumulators
        acc += new_acc

    # Normalize the accumulation by the sum of exponentials
    acc /= tl.exp(logexpsum)

    # Store the result in O
    tl.store(O + b * stride_o_ob + h * stride_o_oh + d * stride_o_od, acc)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 128, 'BLOCK_DMODEL': 64}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 256, 'BLOCK_DMODEL': 128}, num_warps=8),
    ],
    key=['B', 'H', 'Lk', 'D'],
)
@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Pointer to sequence lengths for batches
    Mid_O,     # Pointer to input tensor [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # Pointer to input tensor [batch, head, seq_block_num]
    O,         # Pointer to output tensor [batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,  # Strides for Mid_O
    stride_logexpsum_ob, stride_logexpsum_oh, stride_logexpsum_os,  # Strides for Mid_O_LogExpSum
    stride_o_ob, stride_o_oh, stride_o_od,  # Strides for O
    BLOCK_SEQ: tl.constexpr,  # Block size for sequences
    BLOCK_DMODEL: tl.constexpr  # Block size for model dimensions
):
    # Get program ID
    pid = tl.program_id(axis=0)
    b = pid // (BLOCK_SEQ * BLOCK_DMODEL)
    h = (pid % (BLOCK_SEQ * BLOCK_DMODEL)) // BLOCK_DMODEL
    d = (pid % (BLOCK_SEQ * BLOCK_DMODEL)) % BLOCK_DMODEL

    # Compute the sequence length for this batch
    seqlen = tl.load(B_Seqlen + b)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    logexpsum = tl.zeros((1,), dtype=tl.float32)

    # Iterate over sequence blocks
    for s in range(0, seqlen, BLOCK_SEQ):
        # Load Mid_O and Mid_O_LogExpSum
        mid_o = tl.load(Mid_O + b * stride_mid_ob + h * stride_mid_oh + s * stride_mid_os + d * stride_mid_od, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)
        mid_o_logexpsum = tl.load(Mid_O_LogExpSum + b * stride_logexpsum_ob + h * stride_logexpsum_oh + s * stride_logexpsum_os, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)

        # Compute new values
        new_acc = mid_o * tl.exp(mid_o_logexpsum - logexpsum)
        logexpsum = tl.log(tl.exp(logexpsum) + tl.exp(mid_o_logexpsum))

        # Update accumulators
        acc += new_acc

    # Normalize the accumulation by the sum of exponentials
    acc /= tl.exp(logexpsum)

    # Store the result in O
    tl.store(O + b * stride_o_ob + h * stride_o_oh + d * stride_o_od, acc)

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, BLOCK_SEQ, BLOCK_DMODEL):
    # Get dimensions
    B, H, S, D = Mid_O.shape
    Lk = Mid_O_LogExpSum.shape[2]

    # Check compatibility
    assert Lk == S, "Sequence block number in Mid_O and Mid_O_LogExpSum must match"

    # Compute grid size
    grid = (B * H * D,)

    # Launch the kernel
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ, BLOCK_DMODEL
    )
