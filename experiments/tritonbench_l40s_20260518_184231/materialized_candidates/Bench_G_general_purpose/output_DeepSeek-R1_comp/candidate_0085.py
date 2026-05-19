import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    Mid_O, Mid_O_LogExpSum, O,
    B_Seqlen,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_obsb, stride_mid_osh, stride_mid_oss,
    stride_ob, stride_oh, stride_od,
    head_dim,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Get sequence length for this batch
    b_seqlen = tl.load(B_Seqlen + pid_b)
    seq_block_num = (b_seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    # Compute max logexp across sequence blocks
    max_logexp = -float('inf')
    for s in range(seq_block_num):
        off_logexp = pid_b * stride_mid_obsb + pid_h * stride_mid_osh + s * stride_mid_oss
        current_logexp = tl.load(Mid_O_LogExpSum + off_logexp)
        max_logexp = tl.maximum(max_logexp, current_logexp)
    
    # Compute sum of exponentials
    sum_exp = 0.0
    for s in range(seq_block_num):
        off_logexp = pid_b * stride_mid_obsb + pid_h * stride_mid_osh + s * stride_mid_oss
        current_logexp = tl.load(Mid_O_LogExpSum + off_logexp)
        sum_exp += tl.exp(current_logexp - max_logexp)
    
    # Accumulate the weighted sum into O
    for d in range(0, head_dim, BLOCK_DMODEL):
        d_offsets = d + tl.arange(0, BLOCK_DMODEL)
        mask = d_offsets < head_dim
        
        o_acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
        
        for s in range(seq_block_num):
            # Compute weight for this sequence block
            off_logexp = pid_b * stride_mid_obsb + pid_h * stride_mid_osh + s * stride_mid_oss
            current_logexp = tl.load(Mid_O_LogExpSum + off_logexp)
            weight = tl.exp(current_logexp - max_logexp) / sum_exp
            
            # Load Mid_O data
            off_mid_o = (pid_b * stride_mid_ob + 
                        pid_h * stride_mid_oh + 
                        s * stride_mid_os + 
                        d_offsets * stride_mid_od)
            mid_o = tl.load(Mid_O + off_mid_o, mask=mask, other=0.0)
            
            o_acc += mid_o * weight
        
        # Store the accumulated result to O
        off_o = (pid_b * stride_ob + 
                pid_h * stride_oh + 
                d_offsets * stride_od)
        tl.store(O + off_o, o_acc.to(O.dtype.element_ty), mask=mask)

def flash_decode_stage2(mid_o, mid_o_logexpsum, B_Seqlen, O, BLOCK_SEQ=64, BLOCK_DMODEL=32):
    assert mid_o.dtype == O.dtype, "mid_o and O must have the same dtype"
    assert mid_o.shape[3] == O.shape[2], "head_dim must match between mid_o and O"
    head_dim = mid_o.shape[3]
    batch, num_heads, seq_block_num, _ = mid_o.shape
    
    # Grid for batch and heads
    grid = (batch, num_heads)
    
    # Strides for Mid_O
    stride_mid_ob = mid_o.stride(0)
    stride_mid_oh = mid_o.stride(1)
    stride_mid_os = mid_o.stride(2)
    stride_mid_od = mid_o.stride(3)
    
    # Strides for Mid_O_LogExpSum
    stride_mid_obsb = mid_o_logexpsum.stride(0)
    stride_mid_osh = mid_o_logexpsum.stride(1)
    stride_mid_oss = mid_o_logexpsum.stride(2)
    
    # Strides for O
    stride_ob = O.stride(0)
    stride_oh = O.stride(1)
    stride_od = O.stride(2)
    
    # Launch kernel
    _fwd_kernel_flash_decode_stage2[grid](
        Mid_O=mid_o, Mid_O_LogExpSum=mid_o_logexpsum, O=O,
        B_Seqlen=B_Seqlen,
        stride_mid_ob=stride_mid_ob, stride_mid_oh=stride_mid_oh,
        stride_mid_os=stride_mid_os, stride_mid_od=stride_mid_od,
        stride_mid_obsb=stride_mid_obsb, stride_mid_osh=stride_mid_osh,
        stride_mid_oss=stride_mid_oss,
        stride_ob=stride_ob, stride_oh=stride_oh, stride_od=stride_od,
        head_dim=head_dim,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL
    )
    return O
