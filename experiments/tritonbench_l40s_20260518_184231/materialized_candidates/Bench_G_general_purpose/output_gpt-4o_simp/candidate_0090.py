import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics_ptr, V_ptr, Out_ptr, 
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    batch_size, head_size, seq_len, d_model,
    stride_logics_batch, stride_logics_head, stride_logics_seq,
    stride_v_batch, stride_v_head, stride_v_seq,
    stride_out_batch, stride_out_head, stride_out_seq,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID determines which batch and head we're processing
    pid = tl.program_id(axis=0)
    batch_id = pid // head_size
    head_id = pid % head_size

    # Compute the starting location for this batch and head
    start_loc = tl.load(B_Start_Loc_ptr + batch_id)
    seq_len = tl.load(B_Seqlen_ptr + batch_id)

    # Loop over sequence length in blocks
    for start_n in range(0, seq_len, BLOCK_SIZE):
        offs_n = tl.arange(0, BLOCK_SIZE)
        mask = offs_n < seq_len

        # Compute offsets for Logics, V, and Out
        logics_offset = (
            batch_id * stride_logics_batch +
            head_id * stride_logics_head +
            (start_loc + start_n + offs_n) * stride_logics_seq
        )
        v_offset = (
            batch_id * stride_v_batch +
            head_id * stride_v_head +
            (start_loc + start_n + offs_n) * stride_v_seq
        )
        out_offset = (
            batch_id * stride_out_batch +
            head_id * stride_out_head +
            (start_loc + start_n + offs_n) * stride_out_seq
        )

        # Load data from Logics and V
        logics = tl.load(Logics_ptr + logics_offset, mask=mask, other=-float('inf'))
        v = tl.load(V_ptr + v_offset, mask=mask)

        # Compute softmax normalization
        max_logics = tl.max(logics, axis=0)
        logics = logics - max_logics
        exp_logics = tl.exp(logics)
        sum_exp_logics = tl.sum(exp_logics, axis=0)
        softmax = exp_logics / sum_exp_logics

        # Update output tensor
        out = tl.load(Out_ptr + out_offset, mask=mask)
        out += softmax * v
        tl.store(Out_ptr + out_offset, out, mask=mask)

def token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, batch_size, head_size, seq_len, d_model):
    BLOCK_SIZE = 128  # Example block size, adjust as needed

    # Define grid dimensions
    grid = (batch_size * head_size,)

    # Launch the Triton kernel
    _fwd_kernel[grid](
        Logics, V, Out, 
        B_Loc, B_Start_Loc, B_Seqlen,
        batch_size, head_size, seq_len, d_model,
        Logics.stride(0), Logics.stride(1), Logics.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
