import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_o_b, stride_o_h, stride_o_d,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Determine the current batch and head
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    log_exp_sum = tl.zeros([BLOCK_SEQ], dtype=tl.float32)

    # Loop over sequence blocks
    for seq_block_idx in range(0, B_Seqlen[batch_idx], BLOCK_SEQ):
        # Load Mid_O and Mid_O_LogExpSum for this block
        mid_o_ptr = Mid_O + batch_idx * stride_mid_ob + head_idx * stride_mid_oh + seq_block_idx * stride_mid_os
        mid_o = tl.load(mid_o_ptr + tl.arange(0, BLOCK_DMODEL) * stride_mid_od)
        
        log_exp_sum_ptr = Mid_O_LogExpSum + batch_idx * stride_mid_ob + head_idx * stride_mid_oh + seq_block_idx
        log_exp_sum_block = tl.load(log_exp_sum_ptr + tl.arange(0, BLOCK_SEQ))
        
        # Accumulate weighted values
        acc += mid_o * tl.exp(log_exp_sum_block)
        log_exp_sum += log_exp_sum_block

    # Normalize the accumulated result
    norm_factor = tl.exp(-log_exp_sum)
    result = acc * norm_factor

    # Write back the result to O
    o_ptr = O + batch_idx * stride_o_b + head_idx * stride_o_h
    tl.store(o_ptr + tl.arange(0, BLOCK_DMODEL) * stride_o_d, result)


# Define the PyTorch wrapper function
def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, BLOCK_SEQ, BLOCK_DMODEL):
    # Ensure the dimensions are compatible
    assert Mid_O.size(-1) == BLOCK_DMODEL, "Model dimensions (Lk) must match BLOCK_DMODEL"

    # Get strides
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od = Mid_O.stride()
    stride_o_b, stride_o_h, stride_o_d = O.stride()

    # Launch the Triton kernel
    grid = (B_Seqlen.size(0), Mid_O.size(1))  # Launch a 2D grid for batch and head
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
        stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
        stride_o_b, stride_o_h, stride_o_d,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL
    )

# Example usage
B_Seqlen = torch.tensor([128, 256], dtype=torch.int32)  # Example sequence lengths
Mid_O = torch.rand((2, 4, 32, 64), dtype=torch.float32, device='cuda')  # Example tensor
Mid_O_LogExpSum = torch.rand((2, 4, 32), dtype=torch.float32, device='cuda')  # Example tensor
O = torch.zeros((2, 4, 64), dtype=torch.float32, device='cuda')  # Output tensor

# Define block sizes
BLOCK_SEQ = 32
BLOCK_DMODEL = 64

# Call the wrapper function
flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, BLOCK_SEQ, BLOCK_DMODEL)
