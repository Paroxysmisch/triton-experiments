import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    logics_ptr, 
    b_start_loc_ptr,
    b_seqlen_ptr,
    prob_out_ptr,
    n_batch_head,
    num_heads,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    start_loc = tl.load(b_start_loc_ptr + batch_id)
    seqlen = tl.load(b_seqlen_ptr + batch_id)
    offs = tl.arange(0, BLOCK_SIZE)
    row_idx = start_loc + offs
    mask = offs < seqlen

    logits_offset = (batch_id * num_heads + head_id) * 0 + row_idx  # Adjust if needed for actual layout
    logits = tl.load(logics_ptr + logits_offset, mask=mask, other=-float('inf'))
    max_val = tl.maximum(tl.max(logits, axis=0), -65504.0)
    logits = logits - max_val
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0) + 1e-6
    softmax_vals = exp_logits / sum_exp
    tl.store(prob_out_ptr + logits_offset, softmax_vals, mask=mask)

@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out):
    max_input_len = B_Seqlen.max().item()
    BLOCK_SIZE = 1
    for size in [32, 64, 128, 256, 512, 1024]:
        if size >= max_input_len:
            BLOCK_SIZE = size
            break
    num_warps = 1
    if BLOCK_SIZE > 64:
        num_warps = 2
    if BLOCK_SIZE > 128:
        num_warps = 4
    n_batch = B_Start_Loc.shape[0]
    # Adjust if multiple heads: example with 1 head
    n_heads = 1
    grid = (n_batch * n_heads,)
    logics_ptr = Logics.data_ptr()
    b_start_loc_ptr = B_Start_Loc.data_ptr()
    b_seqlen_ptr = B_Seqlen.data_ptr()
    prob_out_ptr = Prob_Out.data_ptr()
    _fwd_kernel_token_softmax[grid](
        logics_ptr,
        b_start_loc_ptr,
        b_seqlen_ptr,
        prob_out_ptr,
        n_batch,
        n_heads,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1
    )
