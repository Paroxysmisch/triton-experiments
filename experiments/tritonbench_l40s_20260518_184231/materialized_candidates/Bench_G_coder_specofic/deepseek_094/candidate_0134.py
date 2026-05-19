@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, sm_scale_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    stride_Q, stride_K, stride_V, stride_Out,
    BLOCK_M, BLOCK_DMODEL, BLOCK_N,
    batch_size, num_heads, seq_len, Lk,
    **kwargs
):
    pass

def context_attention_fwd(Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen, stride_Q, stride_K, stride_V, stride_Out, batch_size, num_heads, seq_len, Lk):
    # Initialize and dispatch the kernel
    pass
