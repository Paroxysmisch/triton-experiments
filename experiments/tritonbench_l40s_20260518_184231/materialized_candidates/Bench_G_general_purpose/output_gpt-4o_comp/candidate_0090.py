import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_logics_bh, stride_logics_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_out_bh, stride_out_n, stride_out_d,
    BLOCK_N, BLOCK_DMODEL,
    **meta
):
    # Get the program ID for batch and head dimensions
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute the starting location for this batch and head
    b_start_loc = tl.load(B_Start_Loc + pid_b)
    b_seqlen = tl.load(B_Seqlen + pid_b)

    # Define offsets for this batch and head
    offset_logics = pid_b * stride_logics_bh + pid_h * stride_logics_n
    offset_v = pid_b * stride_v_bh + pid_h * stride_v_n
    offset_out = pid_b * stride_out_bh + pid_h * stride_out_n

    # Initialize accumulation and max exponentials
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    e_max = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)

    # Loop over sequence blocks
    for n_start in range(0, b_seqlen, BLOCK_N):
        # Load logits for this block
        logits = tl.load(Logics + offset_logics + n_start, mask=(n_start < b_seqlen))

        # Compute max for numerical stability
        e_max = tl.maximum(e_max, logits)

        # Compute exponentials
        exp_logits = tl.exp(logits - e_max)

        # Load values and compute weighted sum
        values = tl.load(V + offset_v + n_start * stride_v_d, mask=(n_start < b_seqlen))
        acc += exp_logits[:, None] * values

    # Normalize by the sum of exponentials
    sum_exp = tl.sum(exp_logits, axis=0)
    acc /= sum_exp

    # Store result in output
    tl.store(Out + offset_out, acc)

def token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, BLOCK_DMODEL, num_warps=4, num_stages=2):
    # Get dimensions
    batch_size, num_heads, seq_len, _ = Logics.shape

    # Define grid dimensions
    grid = (batch_size, num_heads)

    # Launch the Triton kernel
    triton.kernel(
        _fwd_kernel,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )(
        Logics, V, Out,
        B_Loc, B_Start_Loc, B_Seqlen,
        Logics.stride(0), Logics.stride(1),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_N=seq_len, BLOCK_DMODEL=BLOCK_DMODEL
    )
