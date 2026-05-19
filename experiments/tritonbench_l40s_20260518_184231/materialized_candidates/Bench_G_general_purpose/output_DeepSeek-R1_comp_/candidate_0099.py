import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen_ptr, Mid_O_ptr, Mid_O_LogExpSum_ptr, Out_ptr,
    mid_o_bs, mid_o_hs, mid_o_s, mid_o_ds,
    mid_o_logexpsum_bs, mid_o_logexpsum_hs, mid_o_logexpsum_s,
    out_bs, out_hs, out_ds,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    seq_len = tl.load(B_Seqlen_ptr + batch_idx)
    block_n_size = tl.cdiv(seq_len, BLOCK_SEQ)
    
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    for block_idx in range(block_n_size):
        logexpsum_offset = (batch_idx * mid_o_logexpsum_bs + 
                            head_idx * mid_o_logexpsum_hs + 
                            block_idx * mid_o_logexpsum_s)
        tlogic = tl.load(Mid_O_LogExpSum_ptr + logexpsum_offset)
        
        new_max = tl.maximum(max_logic, tlogic)
        scale = tl.exp(max_logic - new_max)
        sum_exp *= scale
        acc *= scale
        
        exp_current = tl.exp(tlogic - new_max)
        sum_exp += exp_current
        
        mid_o_offset = (batch_idx * mid_o_bs + 
                        head_idx * mid_o_hs + 
                        block_idx * mid_o_s)
        d_offsets = tl.arange(0, BLOCK_DMODEL)
        mid_o_ptrs = Mid_O_ptr + mid_o_offset + d_offsets * mid_o_ds
        tv = tl.load(mid_o_ptrs)
        
        acc += tv * exp_current
        max_logic = new_max
    
    acc = acc / sum_exp
    
    out_offset = batch_idx * out_bs + head_idx * out_hs
    out_ptrs = Out_ptr + out_offset + d_offsets * out_ds
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))

def flash_decode_stage2(B_Seqlen: torch.Tensor, Mid_O: torch.Tensor, 
                        Mid_O_LogExpSum: torch.Tensor, Out: torch.Tensor, 
                        BLOCK_SEQ: int, BLOCK_DMODEL: int):
    assert B_Seqlen.is_cuda and Mid_O.is_cuda and Mid_O_LogExpSum.is_cuda and Out.is_cuda
    assert Mid_O.dim() == 4, "Mid_O should be 4D: (batch, head, block, d_model)"
    assert Mid_O_LogExpSum.dim() == 3, "Mid_O_LogExpSum should be 3D: (batch, head, block)"
    assert Out.dim() == 3, "Out should be 3D: (batch, head, d_model)"
    
    batch_size, head_num = Mid_O.shape[0], Mid_O.shape[1]
    grid = (batch_size, head_num)
    
    mid_o_bs, mid_o_hs, mid_o_s, mid_o_ds = Mid_O.stride()
    mid_o_logexpsum_bs, mid_o_logexpsum_hs, mid_o_logexpsum_s = Mid_O_LogExpSum.stride()
    out_bs, out_hs, out_ds = Out.stride()
    
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, Out,
        mid_o_bs, mid_o_hs, mid_o_s, mid_o_ds,
        mid_o_logexpsum_bs, mid_o_logexpsum_hs, mid_o_logexpsum_s,
        out_bs, out_hs, out_ds,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=3
    )
