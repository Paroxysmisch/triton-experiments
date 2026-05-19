import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Tensor containing the sequence lengths for each batch
    Mid_O,     # Intermediate tensor
    Mid_O_LogExpSum,  # Intermediate tensor for log of sum of exponentials
    O,         # Output tensor
    stride_B_Seqlen_b,  # Stride for B_Seqlen
    stride_Mid_O_b,     # Stride for Mid_O batch dimension
    stride_Mid_O_h,     # Stride for Mid_O head dimension
    stride_Mid_O_s,     # Stride for Mid_O sequence dimension
    stride_Mid_O_d,     # Stride for Mid_O dmodel dimension
    stride_Mid_O_LogExpSum_b,  # Stride for Mid_O_LogExpSum batch dimension
    stride_Mid_O_LogExpSum_h,  # Stride for Mid_O_LogExpSum head dimension
    stride_O_b,         # Stride for O batch dimension
    stride_O_h,         # Stride for O head dimension
    stride_O_d,         # Stride for O dmodel dimension
    BLOCK_SEQ: tl.constexpr,  # Block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr  # Block size for dmodel dimension
):
    # Get program IDs
    pid_b = tl.program_id(axis=0)  # Batch ID
    pid_h = tl.program_id(axis=1)  # Head ID

    # Compute the sequence length for the current batch
    seqlen = tl.load(B_Seqlen + pid_b * stride_B_Seqlen_b)

    # Compute the log of the sum of exponentials for the current batch and head
    log_exp_sum = tl.load(Mid_O_LogExpSum + pid_b * stride_Mid_O_LogExpSum_b + pid_h * stride_Mid_O_LogExpSum_h)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Loop over the sequence dimension
    for s in range(0, seqlen, BLOCK_SEQ):
        # Compute the block size for the current iteration
        block_size = tl.minimum(BLOCK_SEQ, seqlen - s)

        # Load the current block from Mid_O
        mid_o_ptr = Mid_O + pid_b * stride_Mid_O_b + pid_h * stride_Mid_O_h + s * stride_Mid_O_s
        mid_o = tl.load(mid_o_ptr, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)

        # Compute the weighted sum for the current block
        for d in range(0, BLOCK_DMODEL):
            acc[d] += tl.sum(mid_o[:, d] * tl.exp(mid_o[:, d] - log_exp_sum))

    # Normalize the accumulator and store the result in O
    for d in range(0, BLOCK_DMODEL):
        acc[d] /= seqlen

    # Store the result in the output tensor
    o_ptr = O + pid_b * stride_O_b + pid_h * stride_O_h
    tl.store(o_ptr, acc, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 128, 'BLOCK_DMODEL': 64}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 256, 'BLOCK_DMODEL': 128}, num_warps=8),
    ],
    key=['B', 'H', 'D']
)
@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Tensor containing the sequence lengths for each batch
    Mid_O,     # Intermediate tensor
    Mid_O_LogExpSum,  # Intermediate tensor for log of sum of exponentials
    O,         # Output tensor
    stride_B_Seqlen_b,  # Stride for B_Seqlen
    stride_Mid_O_b,     # Stride for Mid_O batch dimension
    stride_Mid_O_h,     # Stride for Mid_O head dimension
    stride_Mid_O_s,     # Stride for Mid_O sequence dimension
    stride_Mid_O_d,     # Stride for Mid_O dmodel dimension
    stride_Mid_O_LogExpSum_b,  # Stride for Mid_O_LogExpSum batch dimension
    stride_Mid_O_LogExpSum_h,  # Stride for Mid_O_LogExpSum head dimension
    stride_O_b,         # Stride for O batch dimension
    stride_O_h,         # Stride for O head dimension
    stride_O_d,         # Stride for O dmodel dimension
    B: tl.constexpr,  # Batch size
    H: tl.constexpr,  # Number of heads
    D: tl.constexpr,  # Dmodel dimension
    BLOCK_SEQ: tl.constexpr,  # Block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr  # Block size for dmodel dimension
):
    # Get program IDs
    pid_b = tl.program_id(axis=0)  # Batch ID
    pid_h = tl.program_id(axis=1)  # Head ID

    # Compute the sequence length for the current batch
    seqlen = tl.load(B_Seqlen + pid_b * stride_B_Seqlen_b)

    # Compute the log of the sum of exponentials for the current batch and head
    log_exp_sum = tl.load(Mid_O_LogExpSum + pid_b * stride_Mid_O_LogExpSum_b + pid_h * stride_Mid_O_LogExpSum_h)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Loop over the sequence dimension
    for s in range(0, seqlen, BLOCK_SEQ):
        # Compute the block size for the current iteration
        block_size = tl.minimum(BLOCK_SEQ, seqlen - s)

        # Load the current block from Mid_O
        mid_o_ptr = Mid_O + pid_b * stride_Mid_O_b + pid_h * stride_Mid_O_h + s * stride_Mid_O_s
        mid_o = tl.load(mid_o_ptr, mask=s + tl.arange(0, BLOCK_SEQ) < seqlen, other=0.0)

        # Compute the weighted sum for the current block
        for d in range(0, BLOCK_DMODEL):
            acc[d] += tl.sum(mid_o[:, d] * tl.exp(mid_o[:, d] - log_exp_sum))

    # Normalize the accumulator and store the result in O
    for d in range(0, BLOCK_DMODEL):
        acc[d] /= seqlen

    # Store the result in the output tensor
    o_ptr = O + pid_b * stride_O_b + pid_h * stride_O_h
    tl.store(o_ptr, acc, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL)

def fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, BLOCK_SEQ=128, BLOCK_DMODEL=64):
    B, H, S, D = Mid_O.shape
    grid = (B, H)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
        B_Seqlen.stride(0), Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1),
        O.stride(0), O.stride(1), O.stride(2),
        B, H, D, BLOCK_SEQ, BLOCK_DMODEL
    )
