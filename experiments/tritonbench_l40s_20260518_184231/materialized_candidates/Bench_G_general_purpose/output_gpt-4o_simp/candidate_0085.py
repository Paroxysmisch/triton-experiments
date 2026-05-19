import triton
import triton.language as tl

# Constants defining the block sizes
BLOCK_SEQ = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
                                    stride_b_seqlen, stride_mid_o, stride_mid_o_logexpsum, stride_o,
                                    n_batch, n_head, d_model):
    pid = tl.program_id(axis=0)
    
    # Calculate the batch and head indices from the program ID
    batch_id = pid // n_head
    head_id = pid % n_head
    
    # Load the sequence length for this batch
    seq_len = tl.load(B_Seqlen + batch_id * stride_b_seqlen)
    
    # Initialize an accumulator for the weighted sum
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    
    # Iterate over sequence blocks
    for seq_start in range(0, seq_len, BLOCK_SEQ):
        # Calculate the effective sequence length for the current block
        seq_block_len = min(BLOCK_SEQ, seq_len - seq_start)
        
        # Load the intermediate results and their log-exp sums
        mid_o_ptr = Mid_O + (batch_id * n_head + head_id) * stride_mid_o + seq_start * d_model
        mid_o_logexpsum_ptr = Mid_O_LogExpSum + (batch_id * n_head + head_id) * stride_mid_o_logexpsum + seq_start
        
        # Load the current block of Mid_O and Mid_O_LogExpSum
        mid_o = tl.load(mid_o_ptr + tl.arange(0, seq_block_len)[:, None] * d_model + tl.arange(0, BLOCK_DMODEL)[None, :])
        mid_o_logexpsum = tl.load(mid_o_logexpsum_ptr + tl.arange(0, seq_block_len))
        
        # Compute the weighted sum
        weights = tl.exp(mid_o_logexpsum[:, None] - mid_o)
        acc += tl.sum(weights * mid_o, axis=0)
    
    # Normalize the accumulator and store the result in the output tensor
    o_ptr = O + (batch_id * n_head + head_id) * stride_o
    tl.store(o_ptr + tl.arange(0, BLOCK_DMODEL), acc / seq_len)

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, n_batch, n_head, d_model):
    # Calculate the strides for each tensor
    stride_b_seqlen = B_Seqlen.stride(0)
    stride_mid_o = Mid_O.stride(0)
    stride_mid_o_logexpsum = Mid_O_LogExpSum.stride(0)
    stride_o = O.stride(0)
    
    # Launch the Triton kernel
    grid = (n_batch * n_head,)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
        stride_b_seqlen, stride_mid_o, stride_mid_o_logexpsum, stride_o,
        n_batch, n_head, d_model
    )
