import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O, 
    Mid_O_LogExpSum, 
    O,  
    out_logexpsum,  
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    # code from above

@triton.jit
def flash_decode_stage2(
    B_Seqlen,
    M_O, 
    M_O_LogExpSum, 
    O,  
    out_logexpsum,  
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    grid: tuple,
    block: tuple):

    batch_size = B_Seqlen.shape[0]
    head_num = M_O.shape[1]

    for bid in range(grid[0]):
        for hid in range(grid[1]):
            _fwd_kernel_flash_decode_stage2[block](
                B_Seqlen[bid],
                M_O[bid, hid], 
                M_O_LogExpSum[bid, hid], 
                O[bid, hid], 
                out_logexpsum[bid, hid], 
                stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
                stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
                stride_obs, stride_oh, stride_od,
                stride_out_logexpsum_b, stride_out_logexpsum_h,
                BLOCK_SEQ, BLOCK_DMODEL)
