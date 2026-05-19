import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Logics_ptr, V_ptr, Out_ptr, B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr, 
                stride_l, stride_v, stride_o, stride_b_loc, stride_b_start_loc, stride_b_seqlen,
                BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Offsets for batch and head
    logics_offset = batch_id * stride_l + head_id * BLOCK_DMODEL
    v_offset = batch_id * stride_v + head_id * BLOCK_DMODEL
    out_offset = batch_id * stride_o + head_id * BLOCK_DMODEL
    b_loc_offset = batch_id * stride_b_loc
    b_start_loc_offset = batch_id * stride_b_start_loc
    b_seqlen_offset = batch_id * stride_b_seqlen

    # Load sequence length
    seqlen = tl.load(B_Seqlen_ptr + b_seqlen_offset)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Iterate over blocks of the sequence
    for i in range(0, seqlen, BLOCK_N):
        loc = tl.load(B_Loc_ptr + b_loc_offset + i)
        start_loc = tl.load(B_Start_Loc_ptr + b_start_loc_offset + i)
        
        # Compute pointers
        logics_ptr = Logics_ptr + logics_offset + loc * BLOCK_DMODEL
        v_ptr = V_ptr + v_offset + start_loc * BLOCK_DMODEL
        
        # Load logits and values
        logits = tl.load(logics_ptr)
        values = tl.load(v_ptr)
        
        # Compute max logits for numerical stability
        max_logits = tl.max(logits, axis=0)
        logits_shifted = logits - max_logits
        
        # Compute exponentials and their sum
        exp_logits = tl.exp(logits_shifted)
        sum_exp += exp_logits
        
        # Accumulate weighted values
        acc += exp_logits * values
    
    # Normalize the accumulated values
    acc /= sum_exp
    
    # Store the result
    out_ptr = Out_ptr + out_offset
    tl.store(out_ptr, acc)

def token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, BLOCK_DMODEL):
    n_batch, n_head, _ = Logics.shape
    BLOCK_N = 128  # Define the block size for the sequence length
    num_warps = 8
    num_stages = 4

    grid = (n_batch, n_head)

    stride_l = Logics.stride(0)
    stride_v = V.stride(0)
    stride_o = Out.stride(0)
    stride_b_loc = B_Loc.stride(0)
    stride_b_start_loc = B_Start_Loc.stride(0)
    stride_b_seqlen = B_Seqlen.stride(0)

    _fwd_kernel[grid](
        Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
        stride_l, stride_v, stride_o, stride_b_loc, stride_b_start_loc, stride_b_seqlen,
        BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps, num_stages=num_stages
    )

# Example usage
torch.manual_seed(42)
n_batch, n_head, seqlen, d_model = 2, 4, 128, 64
Logics = torch.randn((n_batch, n_head, seqlen, d_model), device='cuda', dtype=torch.float32)
V = torch.randn((n_batch, n_head, seqlen, d_model), device='cuda', dtype=torch.float32)
Out = torch.empty((n_batch, n_head, d_model), device='cuda', dtype=torch.float32)
B_Loc = torch.randint(0, seqlen, (n_batch, seqlen), device='cuda', dtype=torch.int32)
B_Start_Loc = torch.randint(0, seqlen, (n_batch, seqlen), device='cuda', dtype=torch.int32)
B_Seqlen = torch.randint(1, seqlen, (n_batch,), device='cuda', dtype=torch.int32)

token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, BLOCK_DMODEL=d_model)
