import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics_ptr, V_ptr, Out_ptr,
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    stride_logics_head, stride_logics_seq, stride_logics_dmodel,
    stride_v_head, stride_v_seq, stride_v_dmodel,
    stride_out_head, stride_out_seq, stride_out_dmodel,
    BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load sequence start and length
    b_start = tl.load(B_Start_Loc_ptr + b_idx)
    seq_len = tl.load(B_Seqlen_ptr + b_idx)

    # Initialize accumulators
    e_max_val = -float('inf')
    e_sum_val = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Loop over sequence in BLOCK_N chunks
    offset_n = 0
    while offset_n < seq_len:
        n_block = tl.arange(0, BLOCK_N)
        n_mask = n_block + offset_n < seq_len

        # Compute the index for logits and v
        idx_logics = b_start + (b_idx * 0)  # placeholder for any additional indexing
        idx_logics += (h_idx * stride_logics_head) + ((offset_n + n_block) * stride_logics_seq)
        logics = tl.load(Logics_ptr + idx_logics, mask=n_mask, other=-float('inf'))

        # Partial max
        block_max = tl.max(logics, axis=0)
        e_max_val = tl.maximum(e_max_val, block_max)

        offset_n += BLOCK_N

    offset_n = 0
    while offset_n < seq_len:
        n_block = tl.arange(0, BLOCK_N)
        n_mask = n_block + offset_n < seq_len

        idx_logics = b_start + (b_idx * 0)
        idx_logics += (h_idx * stride_logics_head) + ((offset_n + n_block) * stride_logics_seq)
        logics = tl.load(Logics_ptr + idx_logics, mask=n_mask, other=-float('inf'))
        logics_shifted = logics - e_max_val
        exp_val = tl.exp(logics_shifted)
        e_sum_val += tl.sum(exp_val, axis=0)

        # Load V and accumulate partial
        idx_v = (b_idx * 0) + (h_idx * stride_v_head) + ((offset_n + n_block) * stride_v_seq)
        acc_block = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
        for d_off in range(0, BLOCK_DMODEL):
            v_val = tl.load(V_ptr + idx_v + d_off * stride_v_dmodel, mask=n_mask, other=0.0)
            acc_block += exp_val * v_val
        acc += acc_block

        offset_n += BLOCK_N

    # Normalize with softmax sum
    inv_e_sum = 1.0 / e_sum_val
    acc *= inv_e_sum

    # Store results to Out
    # We'll do a single pass for storing in blocks of BLOCK_DMODEL
    for d_off in range(0, BLOCK_DMODEL):
        out_idx = (b_idx * 0) + (h_idx * stride_out_head) + (tl.arange(0, 1) * stride_out_seq)
        out_idx += d_off * stride_out_dmodel
        tl.store(Out_ptr + out_idx, acc[d_off])

def token_softmax_reducev_fwd(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    BLOCK_N=128, BLOCK_DMODEL=64,
    num_warps=4, num_stages=2
):
    assert Logics.is_cuda and V.is_cuda and Out.is_cuda
    batch_size = Logics.shape[0]
    heads = Logics.shape[1]
    stride_logics_head = Logics.stride(1)
    stride_logics_seq = Logics.stride(2)
    stride_logics_dmodel = Logics.stride(3)
    stride_v_head = V.stride(1)
    stride_v_seq = V.stride(2)
    stride_v_dmodel = V.stride(3)
    stride_out_head = Out.stride(1)
    stride_out_seq = Out.stride(2)
    stride_out_dmodel = Out.stride(3)

    grid = (batch_size, heads)
    _fwd_kernel[grid](
        Logics, V, Out,
        B_Loc, B_Start_Loc, B_Seqlen,
        stride_logics_head, stride_logics_seq, stride_logics_dmodel,
        stride_v_head, stride_v_seq, stride_v_dmodel,
        stride_out_head, stride_out_seq, stride_out_dmodel,
        BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps, num_stages=num_stages
    )
