import triton
import torch

triton_kernel = triton.get_kernel(triton_fwd_kernel_flash_decode_stage2)

class flash_decode_stage2(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
                stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
                stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
                stride_obs, stride_oh, stride_od,
                stride_out_logexpsum_b, stride_out_logexpsum_h,
                BLOCK_SEQ, BLOCK_DMODEL):
        
        assert O.is_contiguous()
        assert Mid_O.is_contiguous()
        assert Mid_O_LogExpSum.is_contiguous()
        out_logexpsum = O.new_empty(Mid_O_LogExpSum.size()).to(device=O.device)

        batch_size, dim_model = O.size(0), O.size(2)
        
        # Use Triton kernel
        triton. LaunchKernel(
            triton_kernel,
            (batch_size, dim_model),
            out_logexpsum=out_logexpsum,
            Mid_O=Mid_O,
            Mid_O_LogExpSum=Mid_O_LogExpSum,
            O=O,
            B_Seqlen=B_Seqlen,
            stride_mid_ob=stride_mid_ob, stride_mid_oh=stride_mid_oh, stride_mid_os=stride_mid_os, stride_mid_od=stride_mid_od,
            stride_mid_o_eb=stride_mid_o_eb, stride_mid_o_eh=stride_mid_o_eh, stride_mid_o_es=stride_mid_o_es,
            stride_obs=stride_obs, stride_oh=stride_oh, stride_od=stride_od,
            stride_out_logexpsum_b=stride_out_logexpsum_b, stride_out_logexpsum_h=stride_out_logexpsum_h,
            BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL
        )
        return O
