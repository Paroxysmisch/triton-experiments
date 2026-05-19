import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics_ptr,  # Input logits tensor
    V_ptr,       # Input value tensor
    Out_ptr,     # Output tensor
    B_Loc_ptr,   # Auxiliary tensor for batch location
    B_Start_Loc_ptr,  # Auxiliary tensor for batch start location
    B_Seqlen_ptr,  # Auxiliary tensor for sequence length
    stride_logic_h,  # Stride for logits in head dimension
    stride_logic_s,  # Stride for logits in sequence dimension
    stride_v_h,      # Stride for V in head dimension
    stride_v_s,      # Stride for V in sequence dimension
    stride_v_d,      # Stride for V in feature dimension
    stride_out_h,    # Stride for Out in head dimension
    stride_out_s,    # Stride for Out in sequence dimension
    stride_out_d,    # Stride for Out in feature dimension
    batch_size,      # Batch size
    head_size,       # Head size
    seq_len,         # Sequence length
    feature_dim,     # Feature dimension
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID in the grid
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Compute the start and end indices for the batch and head
    start_n = B_Start_Loc_ptr[bid]
    end_n = start_n + B_Seqlen_ptr[bid]

    # Compute the offset for the current batch and head
    offs_h = pid * stride_logic_h
    offs_s = tl.arange(0, BLOCK_SIZE) * stride_logic_s

    # Initialize the output and max value
    out = tl.zeros((BLOCK_SIZE, feature_dim), dtype=tl.float32)
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    # Loop over the sequence length
    for n in range(start_n, end_n, BLOCK_SIZE):
        # Compute the current sequence offset
        offs_n = n + tl.arange(0, BLOCK_SIZE) * stride_logic_s

        # Load the logits and values
        logits = tl.load(Logics_ptr + offs_h + offs_n, mask=offs_n < end_n, other=float('-inf'))
        values = tl.load(V_ptr + offs_h + offs_n * stride_v_s, mask=offs_n < end_n, other=0.0)

        # Compute the max value
        max_val = tl.maximum(max_val, logits)

        # Compute the exponentials
        exp_logits = tl.exp(logits - max_val)

        # Compute the weighted sum
        out += exp_logits[:, None] * values

    # Normalize the output
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    out /= sum_exp_logits[:, None]

    # Store the result
    tl.store(Out_ptr + offs_h + start_n * stride_out_s, out, mask=offs_n < end_n)

import torch

def token_softmax_reducev_fwd(
    Logics,  # Input logits tensor
    V,       # Input value tensor
    B_Loc,   # Auxiliary tensor for batch location
    B_Start_Loc,  # Auxiliary tensor for batch start location
    B_Seqlen,  # Auxiliary tensor for sequence length
    batch_size,  # Batch size
    head_size,   # Head size
    seq_len,     # Sequence length
    feature_dim  # Feature dimension
):
    # Create the output tensor
    Out = torch.empty_like(V)

    # Define the grid size
    grid = (head_size, batch_size)

    # Launch the kernel
    _fwd_kernel[grid](
        Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
        Logics.stride(1), Logics.stride(2),
        V.stride(1), V.stride(2), V.stride(3),
        Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, head_size, seq_len, feature_dim,
        BLOCK_SIZE=128
    )

    return Out
